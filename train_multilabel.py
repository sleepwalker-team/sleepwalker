#!/bin/env python3

from functools import partial
import os

import torch
from torch.utils.data import DataLoader, RandomSampler
from torchinfo import summary

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets import ChannelConfig, Ruhrlandklinik
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.utils import get_edf_files_in_repo, kfold_split
from sleepwalker.models import MetaModel, MetaModelEntry, SleepTransformer
from sleepwalker.models.UTime import UTime
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.trainer.utils import append_to_jsonl
from sleepwalker.utils import MlflowSink, logger

# TODO METAMODEL
#   ALLOW DIFFERENT INPUT SIZES/SAMPLE RATES
#   ADD A POST-EMBEDDING MODEL (NOT FC)

edf_folder = "/home/sleepwalker/data/ruhrlandklinik/raw/train-test-2023/"
n_splits = 3
batch_size = 128
epochs = 100
sample_frequency = 100
total_input = "630s"
target_resolution = "30s"
n_samples = None
experiment_name = "ruhrland_metamodel_multilabel"

all_input_channels = [
    "C4-M1",
    "E2-M1",
    "Chest",
    "Abdomen",
    "Saturation",
    "Pulse Waveform",
]

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
        "labels": ["apnea", "hypopnea", "regular breathing"],
        "default": "regular breathing",
        "percentage": 0.5,
    },
    "arousal": {
        "labels": ["arousal", "no arousal"],
        "default": "no arousal",
        "percentage": 0.5,
    },
    "desaturation": {
        "labels": ["desaturation", "no desaturation"],
        "default": "no desaturation",
        "percentage": 0.5,
    },
    "sleep staging": {
        "labels": ["n1", "n2", "n3", "rem", "wake"],
        "default": None,
        "percentage": 0.5,
    },
}

resp_embedding_size = 32
sleep_embedding_size = 32


def filter_patient(edf_path):
    meta = read_edf_meta(edf_path)
    return all(channel in meta["signals"] for channel in all_input_channels)


def build_loader(patients):
    dataset = Ruhrlandklinik(
        channels=[ChannelConfig(name=c, normalizer=None) for c in all_input_channels],
        patients=patients,
        num_workers=8,
        sample_frequency=sample_frequency,
        event_mapping=event_mapping,
        get_item=partial(MultiLabelTrainer.get_item, task_config=task_config),
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

def build_model(ts_len, classes):
    respiratory_model = UTime(
        ts_len=ts_len,
        n_channels=4,
        classes=[f"resp_{i}" for i in range(resp_embedding_size)],
        sampling_frequency="0.01s",
        channel=[16, 32, 64, 128],
        maxpool=[10, 8, 6, 4],
        kernel=[5, 5, 5, 5],
        norm="channel",
        mlp_size=64,
    )
    sleep_model = SleepTransformer(
        classes=[f"sleep_{i}" for i in range(sleep_embedding_size)],
        n_channels=1,
    )

    model = MetaModel(
        classes=classes,
        input_channels=all_input_channels,
        models=[
            MetaModelEntry("respiratory", respiratory_model, ["Chest", "Abdomen", "Saturation", "Pulse Waveform"]),
            MetaModelEntry("sleep staging", sleep_model, ["C4-M1"]),
        ],
        sample_frequency=sample_frequency,
        ts_len=ts_len
    )
    return model

all_patients = get_edf_files_in_repo(edf_folder, recursive=False)
all_patients  = [p for p in all_patients if filter_patient(p)]
all_patients = all_patients[:10]

if os.path.exists("sleepwalker.log"):
    os.remove("sleepwalker.log")

logger.add_sink(MlflowSink(tracking_uri="sqlite:///mlflow.sqlite", experiment="MyExperiment"))

trainer = MultiLabelTrainer(
    epochs=epochs,
    optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
    lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=1e-2, total_iters=50
    ),
    task_config=task_config,
    save_every=10,
    loss_function=torch.nn.functional.cross_entropy,
)

for i, (train_patients, test_patients) in enumerate(kfold_split(all_patients, n_splits=n_splits)):
    logger.start_run(run_name=f"XVAL {i}")

    train_loader, dataset = build_loader(train_patients)
    missing = sorted(set(dataset.get_classes()) - set(trainer.classes))
    if len(missing) > 0:
        raise ValueError(f"Task config is missing dataset classes: {missing}")

    model = build_model(dataset.get_timeseries_len(), trainer.classes)
    summary(model, input_size=(1, dataset.get_timeseries_len(), len(all_input_channels)), depth=6, row_settings=["hide_recursive_layers"])

    train_dict = trainer.fit(model, train_loader)
    if "checkpoint" in train_dict:
        state_dict = torch.load(os.path.join(train_dict["checkpoint"], "model.pt"), map_location="cpu")
        model.load_state_dict(state_dict)

    test_loader, _ = build_loader(test_patients)
    test_loss, test_cm = trainer.test(model, test_loader)

    record = {
        "xval": i,
        "test_loss": test_loss,
        "test_cm": test_cm,
        "train_loss": train_dict["losses"],
        "train_cm": train_dict["cms"],
        "classes": trainer.classes,
        "task_config": task_config,
        "input_channels": all_input_channels,
    }
    if "best_model" in train_dict:
        record["best_model"] = train_dict["best_model"]
    append_to_jsonl(experiment_name, record)
    logger.end_run()
