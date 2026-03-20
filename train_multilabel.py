#!/bin/env python3

from functools import partial
import os

import torch
from torch.utils.data import DataLoader, RandomSampler
from torchinfo import summary

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets import ChannelConfig, Ruhrlandklinik
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.utils import get_edf_files_in_repo, kfold_split
from sleepwalker.models import MetaModel, MetaModelEntry, SleepTransformer
from sleepwalker.models.UTime import UTime
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.trainer.utils import append_to_jsonl
from sleepwalker.utils import MlflowSink, logger

# TODO METAMODEL
#   ADD A POST-EMBEDDING MODEL (NOT FC)
#   ALLOW MULTIPLE INPUT RESOLUTIONS?!?!

edf_folder = "/raid/sleepwalker/ruhrlandklinik/raw/train-test-2023"
edf_folder_test = "/raid/sleepwalker/ruhrlandklinik/raw/val-2024"
n_splits = 3
batch_size = 128
epochs = 100
sample_frequency = 100
total_input = "630s"
target_resolution = "30s"
n_samples = None
experiment_name = "ruhrland_metamodel_multilabel"

# TODO FORM HERE
channels_cfgs = [
    ChannelConfig(name=c, normalizer=EEGFilterNormalizer(fs = sample_frequency), group="EEG") for c in ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"]
] + [
    ChannelConfig(name=c, normalizer=None, group=None) for c in ["Chest", "Abdomen", "Saturation","Pulse Waveform"]
]

all_input_channels = ["EEG", "Chest", "Abdomen", "Saturation","Pulse Waveform"]

def filter_patient(edf_path):
    meta = read_edf_meta(edf_path)
    if any(c in meta["signals"] for c in ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"]):
        return all(channel in meta["signals"] for channel in ["Chest", "Abdomen", "Saturation","Pulse Waveform"])
    else: 
        return False

event_mapping = {
    "wach": "wake",
    "n1": "n1",
    "n2": "n2",
    "n3": "n3",
    "rem": "rem",
    "a. gemischt": "apnea",
    "a. obstruktiv": "apnea",
    "a. zentral": "apnea",
    "apnoe": "apnea",
    "cheyne stokes": "apnea",
    "fg apnoe geschlossen": "apnea",
    "fg apnoe geöffnet": "apnea",
    "fg-apnoe unbekannt": "apnea",
    "h. obstruktiv": "apnea",
    "h. zentral": "apnea",
    "fg hypopnoe": "hypopnea",
    "hypopnea-gemischt": "hypopnea",
    "hypopnoe": "hypopnea",
    "arousal": "arousal",
    "plm-arousal": "arousal",
    "rera": "arousal",
    "entsättigung": "desaturation",
}

task_config = {
    "breathing": {
        "labels": ["apnea", "hypopnea", "regular"],
        "default": "regular",
        "percentage": 0.5,
        "target_resolution": "10s",
    },
    "arousal": {
        "labels": ["arousal", "no arousal"],
        "default": "no arousal",
        "percentage": 0.5,
        "target_resolution": "1s",
    },
    "desat": {
        "labels": ["desaturation", "no desaturation"],
        "default": "no desaturation",
        "percentage": 0.5,
        "target_resolution": "10s",
    },
    "sleep": {
        "labels": ["n1", "n2", "n3", "rem", "wake"],
        "default": None,
        "percentage": 0.5,
        "target_resolution": "30s",
    },
}
normalized_task_config = MultiLabelTrainer.normalize_task_config(task_config)

def build_loader(patients):
    dataset = Ruhrlandklinik(
        channels=channels_cfgs,
        patients=patients,
        num_workers=8,
        sample_frequency=sample_frequency,
        event_mapping=event_mapping,
        get_target=partial(MultiLabelTrainer.get_target, task_config=normalized_task_config),
        total_input=total_input,
        target_resolution=target_resolution,
    )

    sampler = RandomSampler(dataset, num_samples=n_samples) if n_samples is not None else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=batch_collate,
        drop_last=False,
        persistent_workers=True,
    )
    return loader, dataset

def build_model(ts_len):
    respiratory_model = UTime(
        ts_len=ts_len,
        n_channels=4,
        classes=None,
        sampling_frequency="0.01s", # TODO CHANGE
        channel=[16, 32, 64, 128],
        maxpool=[10, 8, 6, 4],
        kernel=[5, 5, 5, 5],
        norm="channel",
        mlp_size=64,
    )
    sleep_model = SleepTransformer(
        classes=None,
        n_channels=1,
    )

    model = MetaModel(
        task_config=normalized_task_config,
        input_channels=all_input_channels, # TODO WHY?
        models=[
            MetaModelEntry(respiratory_model, ["Chest", "Abdomen", "Saturation", "Pulse Waveform"]),
            MetaModelEntry(sleep_model, ["EEG"]),
        ],
    )
    return model

train_patients = get_edf_files_in_repo(edf_folder, recursive=True)
train_patients  = [p for p in train_patients if filter_patient(p)]

test_patients = get_edf_files_in_repo(edf_folder, recursive=True)
test_patients  = [p for p in test_patients if filter_patient(p)]
# all_patients = all_patients[:10]

if os.path.exists("sleepwalker.log"):
    os.remove("sleepwalker.log")

logger.add_sink(MlflowSink(tracking_uri="sqlite:///mlflow.sqlite", experiment="MyExperiment"))

trainer = MultiLabelTrainer(
    epochs=epochs,
    optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
    lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=1e-2, total_iters=50
    ),
    task_config=normalized_task_config,
    save_every=10,
    loss_function=torch.nn.functional.cross_entropy,
    log_batches=False
)

train_loader, dataset = build_loader(train_patients)
missing = sorted(set(dataset.get_classes()) - set(trainer.classes))
if len(missing) > 0:
    raise ValueError(f"Task config is missing dataset classes: {missing}")

model = build_model(dataset.get_timeseries_len())
summary(model, input_size=(1, dataset.get_timeseries_len(), len(channels_cfgs)), depth=6, row_settings=["hide_recursive_layers"])

train_dict = trainer.fit(model, train_loader)
if "checkpoint" in train_dict:
    state_dict = torch.load(os.path.join(train_dict["checkpoint"], "model.pt"), map_location="cpu")
    model.load_state_dict(state_dict)

test_loader, _ = build_loader(test_patients)
test_loss, test_cm = trainer.test(model, test_loader)

record = {
    "test_loss": test_loss,
    "test_cm": test_cm,
    "train_loss": train_dict["losses"],
    "train_cm": train_dict["cms"],
    "classes": trainer.classes,
    "task_config": normalized_task_config,
    "input_channels": all_input_channels,
}
if "best_model" in train_dict:
    record["best_model"] = train_dict["best_model"]
append_to_jsonl(experiment_name, record)
logger.end_run()
