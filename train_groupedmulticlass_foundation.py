#!/bin/env python3

from functools import partial
import logging
import os
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader

from torchinfo import summary
from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets import ChannelConfig
from sleepwalker.datasets.ABC import ABC
from sleepwalker.datasets.Apples import Apples
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.CAP import CAP
from sleepwalker.datasets.ISRUC import ISRUC
from sleepwalker.datasets.MNC import MNC
from sleepwalker.datasets.MROS import MROS
from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.NCHSDB import NCHSDB
from sleepwalker.datasets.SHHS import SHHS
from sleepwalker.datasets.SVUH_UCD import SVUH_UCD
from sleepwalker.datasets.SleepEDFx import SleepEDFx
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.utils import get_edf_files_in_repo, kfold_split, random_split
from sleepwalker.datasets import Ruhrlandklinik

from sleepwalker.models import SleepTransformer
from sleepwalker.trainer.GroupedChanelMulticlassTrainer import GroupedChanelMulticlassTrainer
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.utils import append_to_jsonl
from sleepwalker.utils import logger, MlflowSink, suppress_stdout_logging

import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')

batch_size = 512
# batch_size = 196
epochs = 200
total_input = "630s"
target_resolution = "30s"
n_samples = 200_000
experiment_name = "grouped_transformer"
num_workers_dataset = 12
num_workers_dataloader = 12
sample_frequency = 100
preload_windows = 30_000

groups = [[
    "F3", "F4", "C3", "C4", "O1", "O2", "M1", "M2",  # ABC, MNC, MROS, Ruhrland
    "C3_M2", "C4_M1", "O2_M1", "O1_M2", # Apples
    "F4-C4", "P4-O2", "C4-P4", "C4-A1", # CAP
    "F3-M2", "C3-M2", "O1-M2", "F4-M1", "C4-M1", "O2-M1", # ISRUC
    "EEG", # SHHS
    "EEG C3-M2", "EEG O2-M1", "EEG O1-M2", "EEG F3-M2", "EEG C4-M1", "EEG F4-M1", #NHCSDB
    "EEG Fpz-Cz", "EEG Pz-Oz", #SleepEDFx
    "C3A2", "C4A1", # SVUH-UCD
]]

# logger.add_sink(MlflowSink(tracking_uri="sqlite:///mlflow.sqlite", experiment=experiment_name))

with open(os.path.expanduser("~/mlflow/auth_config.ini"),"r") as f:
    TRACKING_URI=f.read().strip()
ARTIFACT_URI="/raid/mlruns"

logger.add_sink(MlflowSink(tracking_uri=TRACKING_URI, experiment=experiment_name, artifact_uri=ARTIFACT_URI))
logger.start_run(run_name=experiment_name)

# TRAIN / VAL
def build_abc(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True) #[:10] # XXX
    train_patients, test_patients = random_split(edf_files, test_frac=0.1)
    
    def _create(patients):
        channels = [
            ChannelConfig(name="F3", normalizer=EEGFilterNormalizer(fs = sample_frequency)),
            ChannelConfig(name="F4", normalizer=EEGFilterNormalizer(fs = sample_frequency)),
            ChannelConfig(name="C3", normalizer=EEGFilterNormalizer(fs = sample_frequency)),
            ChannelConfig(name="C4", normalizer=EEGFilterNormalizer(fs = sample_frequency)),
            ChannelConfig(name="O1", normalizer=EEGFilterNormalizer(fs = sample_frequency)),
            ChannelConfig(name="O2", normalizer=EEGFilterNormalizer(fs = sample_frequency)),
            ChannelConfig(name="M1", normalizer=EEGFilterNormalizer(fs = sample_frequency)),
            ChannelConfig(name="M2", normalizer=EEGFilterNormalizer(fs = sample_frequency)),
        ]

        event_mapping = {
            "wake|0": "wake",
            "stage 1 sleep|1": "n1",
            "stage 2 sleep|2": "n2",
            "stage 3 sleep|3": "n3",
            "rem sleep|5": "rem"
        }
        
        return ABC(
            annotator="nsrr",
            channels = channels,
            patients = patients,
            num_workers = num_workers_dataset,
            sample_frequency = sample_frequency,
            event_mapping = event_mapping,
            get_item = partial(GroupedChanelMulticlassTrainer.get_item, groups=groups),
            preload_windows = preload_windows,
            total_input = total_input, 
            target_resolution = target_resolution,
        )

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_ds = _create(train_patients)
        logger.uncontext()

        logger.context("VAL")
        test_ds = _create(test_patients)
        logger.uncontext()

    logger.info(f"Loaded {train_ds.get_n_patients()} / {len(train_patients)} patients for training and {test_ds.get_n_patients()} / {len(test_patients)} patients for validation")
    return train_ds, test_ds

def build_apples(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    train_patients, test_patients = random_split(edf_files, test_frac=0.1)
    
    def _create(patients):
        channels = [
            ChannelConfig(name="C3_M2", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="C4_M1", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="O2_M1", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="O1_M2", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
        ]

        event_mapping = {
            "N1":"n1",
            "N2":"n2",
            "N3":"n3",
            "R":"rem",
            "W":"wake"
        }

        return Apples(
            channels = channels,
            patients = patients,
            num_workers = num_workers_dataset,
            sample_frequency = sample_frequency,
            event_mapping = event_mapping,
            get_item = partial(GroupedChanelMulticlassTrainer.get_item, groups=groups),
            preload_windows = preload_windows, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_ds = _create(train_patients)
        logger.uncontext()

        logger.context("VAL")
        test_ds = _create(test_patients)
        logger.uncontext()
    logger.info(f"Loaded {train_ds.get_n_patients()} / {len(train_patients)} patients for training and {test_ds.get_n_patients()} / {len(test_patients)} patients for validation")
    return train_ds, test_ds

def build_cap(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    train_patients, test_patients = random_split(edf_files, test_frac=0.1)
    
    def _create(patients):
        channels = [
            ChannelConfig(name="F4-C4", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="P4-O2", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="C4-P4", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="C4-A1", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
        ]

        event_mapping = {
            "S1":"n1",
            "S2":"n2",
            "S3":"n3",
            "S4":"n3",
            "R":"rem",
            "W":"wake"
        }

        return CAP(
            channels = channels,
            patients = patients,
            num_workers = num_workers_dataset,
            sample_frequency = sample_frequency,
            event_mapping = event_mapping,
            get_item = partial(GroupedChanelMulticlassTrainer.get_item, groups=groups),
            preload_windows = preload_windows, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_ds = _create(train_patients)
        logger.uncontext()

        logger.context("VAL")
        test_ds = _create(test_patients)
        logger.uncontext()

    logger.info(f"Loaded {train_ds.get_n_patients()} / {len(train_patients)} patients for training and {test_ds.get_n_patients()} / {len(test_patients)} patients for validation")
    return train_ds, test_ds

def build_isruc(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    train_patients, test_patients = random_split(edf_files, test_frac=0.1)
    
    def _create(patients):
        channels = [
            ChannelConfig(name="F3-M2", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="C3-M2", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="O1-M2", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="F4-M1", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="C4-M1", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="O2-M1", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
        ]

        event_mapping = {
            "N1":"n1",
            "N2":"n2",
            "n2":"n2",
            "N3":"n3",
            "R":"rem",
            "W":"wake",
            "w":"wake"
        }

        return ISRUC(
            merge = True,
            channels = channels,
            patients = patients,
            num_workers = num_workers_dataset,
            sample_frequency = sample_frequency,
            event_mapping = event_mapping,
            get_item = partial(GroupedChanelMulticlassTrainer.get_item, groups=groups),
            preload_windows = preload_windows, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_ds = _create(train_patients)
        logger.uncontext()

        logger.context("VAL")
        test_ds = _create(test_patients)
        logger.uncontext()

    logger.info(f"Loaded {train_ds.get_n_patients()} / {len(train_patients)} patients for training and {test_ds.get_n_patients()} / {len(test_patients)} patients for validation")
    return train_ds, test_ds

def build_mnc(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    train_patients, test_patients = random_split(edf_files, test_frac=0.1)
    
    def _create(patients):
        channels = [
            ChannelConfig(name="F3", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="F4", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="C3", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="C4", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
        ]

        event_mapping = {
            "nrem1":"n1",
            "nrem2":"n2",
            "nrem3":"n3",
            "rem":"rem",
            "wake":"wake",
        }

        return MNC(
            channels = channels,
            patients = patients,
            num_workers = num_workers_dataset,
            sample_frequency = sample_frequency,
            event_mapping = event_mapping,
            get_item = partial(GroupedChanelMulticlassTrainer.get_item, groups=groups),
            preload_windows = preload_windows, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_ds = _create(train_patients)
        logger.uncontext()

        logger.context("VAL")
        test_ds = _create(test_patients)
        logger.uncontext()

    logger.info(f"Loaded {train_ds.get_n_patients()} / {len(train_patients)} patients for training and {test_ds.get_n_patients()} / {len(test_patients)} patients for validation")
    return train_ds, test_ds

def build_nchsdb(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    train_patients, test_patients = random_split(edf_files, test_frac=0.1)
    
    def _create(patients):
        channels = [
            ChannelConfig(name="EEG C3-M2", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="EEG O2-M1", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="EEG O1-M2", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="EEG F3-M2", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="EEG C4-M1", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="EEG F4-M1 ", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
        ]

        event_mapping = {
            "Sleep stage 1":"n1",
            "Sleep stage 2":"n2",
            "Sleep stage 3":"n3",
            "Sleep stage R":"rem",
            "Sleep stage W":"wake",
        }

        return NCHSDB(
            channels = channels,
            patients = patients,
            num_workers = num_workers_dataset,
            sample_frequency = sample_frequency,
            event_mapping = event_mapping,
            get_item = partial(GroupedChanelMulticlassTrainer.get_item, groups=groups),
            preload_windows = preload_windows, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_ds = _create(train_patients)
        logger.uncontext()

        logger.context("VAL")
        test_ds = _create(test_patients)
        logger.uncontext()

    logger.info(f"Loaded {train_ds.get_n_patients()} / {len(train_patients)} patients for training and {test_ds.get_n_patients()} / {len(test_patients)} patients for validation")
    return train_ds, test_ds

def build_mros(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    train_patients, test_patients = random_split(edf_files, test_frac=0.1)
    
    def _create(patients):
        channels = [
            ChannelConfig(name="C3", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ChannelConfig(name="C4", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
        ]

        event_mapping = {
            "wake|0": "wake",
            "stage 1 sleep|1": "n1",
            "stage 2 sleep|2": "n2",
            "stage 3 sleep|3": "n3",
            "stage 4 sleep|4": "n3",
            "rem sleep|5": "rem"
        }

        return MROS(
            channels = channels,
            patients = patients,
            num_workers = num_workers_dataset,
            sample_frequency = sample_frequency,
            event_mapping = event_mapping,
            get_item = partial(GroupedChanelMulticlassTrainer.get_item, groups=groups),
            preload_windows = preload_windows, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_ds = _create(train_patients)
        logger.uncontext()

        logger.context("VAL")
        test_ds = _create(test_patients)
        logger.uncontext()

    logger.info(f"Loaded {train_ds.get_n_patients()} / {len(train_patients)} patients for training and {test_ds.get_n_patients()} / {len(test_patients)} patients for validation")
    return train_ds, test_ds

# TEST

def build_sleepedfx(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)# [:10] # XXX

    with suppress_stdout_logging(logger):
        dataset = SleepEDFx(
            channels = [
                ChannelConfig(name="EEG Fpz-Cz", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
                ChannelConfig(name="EEG Pz-Oz", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ],
            patients = edf_files,
            num_workers = num_workers_dataset,
            sample_frequency = sample_frequency,
            event_mapping = {   
                "sleep stage w": "wake",
                "sleep stage 1": "n1",
                "sleep stage 2": "n2",
                "sleep stage 3": "n3",
                "sleep stage 4": "n3",
                "sleep stage r": "rem"
            },
            get_item = partial(GroupedChanelMulticlassTrainer.get_item, groups=groups),
            preload_windows = preload_windows, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    logger.info(f"Loaded {dataset.get_n_patients()} / {len(edf_files)} patients for testing")

    return dataset

def build_svuh_ucd(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True, ending="rec")

    with suppress_stdout_logging(logger):
        dataset = SVUH_UCD(
            channels = [
                ChannelConfig(name="C3A2", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
                ChannelConfig(name="C4A1", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ],
            patients = edf_files,
            num_workers = num_workers_dataset,
            sample_frequency = sample_frequency,
            event_mapping = {   
                    "0": "wake",
                    "1": "rem",
                    "2": "n1",
                    "3": "n2",
                    "4": "n3",
                    "5": "n3"
            },
            get_item = partial(GroupedChanelMulticlassTrainer.get_item, groups=groups),
            preload_windows = preload_windows, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    logger.info(f"Loaded {dataset.get_n_patients()} / {len(edf_files)} patients for testing")
    return dataset

def build_ruhrland(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)

    with suppress_stdout_logging(logger):
        dataset = Ruhrlandklinik(
            channels = [
                ChannelConfig(name="F3", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
                ChannelConfig(name="F4", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
                ChannelConfig(name="C3", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
                ChannelConfig(name="C4", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
                ChannelConfig(name="O1", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
                ChannelConfig(name="O2", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
                ChannelConfig(name="M1", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
                ChannelConfig(name="M2", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ],
            return_nox = False,
            patients = edf_files,
            num_workers = num_workers_dataset,
            sample_frequency = sample_frequency,
            event_mapping = {   
                "wach": "wake",
                "n1": "n1",
                "n2": "n2",
                "n3": "n3",
                "rem": "rem"
            },
            get_item = partial(GroupedChanelMulticlassTrainer.get_item, groups=groups), 
            preload_windows = preload_windows, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    logger.info(f"Loaded {dataset.get_n_patients()} / {len(edf_files)} patients for testing")
    return dataset

def build_shhs(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)

    with suppress_stdout_logging(logger):
        dataset = SHHS(
            channels = [
                ChannelConfig(name="EEG", normalizer=EEGFilterNormalizer(fs=sample_frequency)),
            ],
            patients = edf_files,
            num_workers = num_workers_dataset,
            sample_frequency = sample_frequency,
            event_mapping = {
                "wake|0": "wake",
                "stage 1 sleep|1": "n1",
                "stage 2 sleep|2": "n2",
                "stage 3 sleep|3": "n3",
                "stage 4 sleep|4": "n3",
                "rem sleep|5": "rem"
            },
            get_item = partial(GroupedChanelMulticlassTrainer.get_item, groups=groups), 
            preload_windows = preload_windows, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    logger.info(f"Loaded {dataset.get_n_patients()} / {len(edf_files)} patients for testing")
    return dataset

ds = []
logger.context("ABC")
ds.append(build_abc("/raid/sleepwalker/abc"))
logger.uncontext()

logger.context("Apples")
ds.append(build_apples("/raid/sleepwalker/apples/polysomnography"))
logger.uncontext()

logger.context("CAP")
ds.append(build_cap("/raid/sleepwalker/cap"))
logger.uncontext()

logger.context("ISRUC")
ds.append(build_isruc("/raid/sleepwalker/isruc"))
logger.uncontext()

logger.context("MNC")
ds.append(build_mnc("/raid/sleepwalker/mnc/cnc"))
logger.uncontext()

logger.context("NCHSDB")
ds.append(build_nchsdb("/raid/sleepwalker/nchsdb/sleep_data"))
logger.uncontext()

logger.context("MROS")
ds.append(build_mros("/raid/sleepwalker/mros"))
logger.uncontext()

train_ds = [d[0] for d in ds]
val_ds = [d[1] for d in ds]

test_ds = []
test_ds_names = []

logger.context("SleepEDFx")
logger.context("TEST")
test_ds.append(build_sleepedfx("/raid/sleepwalker/sleep-edfx"))
logger.uncontext()
test_ds_names.append("SleepEDFx")
logger.uncontext()

logger.context("Ruhrland")
logger.context("TEST")
test_ds.append(build_ruhrland("/raid/sleepwalker/ruhrlandklinik/raw"))
test_ds_names.append("Ruhrland")
logger.uncontext()
logger.uncontext()

logger.context("SVUH-UCD")
logger.context("TEST")
test_ds.append(build_svuh_ucd("/raid/sleepwalker/svuh-ucd"))
test_ds_names.append("SVUH-UCD")
logger.uncontext()
logger.uncontext()

logger.context("SHHS")
logger.context("TEST")
test_ds.append(build_shhs("/raid/sleepwalker/shhs"))
test_ds_names.append("SHHS")
logger.uncontext()
logger.uncontext()

if os.path.exists("sleepwalker.log"):
    os.remove("sleepwalker.log")

train_multi_ds = MultiDataset(train_ds)
logger.info(f"Loaded {train_multi_ds.n_patients()} for training")

val_multi_ds = MultiDataset(val_ds)
logger.info(f"Loaded {val_multi_ds.n_patients()} for validation")

test_multi_ds = MultiDataset(test_ds)
logger.info(f"Loaded {test_multi_ds.n_patients()} for testing")

train_sampler = RandomSampler(train_multi_ds, num_samples = n_samples) if n_samples is not None else None
train_loader = DataLoader(train_multi_ds, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=num_workers_dataloader, collate_fn=partial(batch_collate, ignore_list=["time", "patient", "data"]), drop_last=False, persistent_workers=True, prefetch_factor=2, pin_memory=True) ### prefetch_factor=12

val_sampler = RandomSampler(val_multi_ds, num_samples = n_samples) if n_samples is not None else None
val_loader = DataLoader(val_multi_ds, batch_size=batch_size, shuffle=val_sampler is None, sampler=val_sampler, num_workers=num_workers_dataloader, collate_fn=partial(batch_collate, ignore_list=["time", "patient", "data"]), drop_last=False, persistent_workers=True, prefetch_factor=2, pin_memory=True)

model = SleepTransformer(classes=train_multi_ds.get_classes(), n_channels=len(groups))
model_stats = summary(model, input_size=(1, train_multi_ds.get_timeseries_len(), len(groups)), depth=5, row_settings=["hide_recursive_layers"])

trainer = GroupedChanelMulticlassTrainer(
    groups=groups,
    epochs=epochs, 
    optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
    lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=50),
    classes = train_multi_ds.get_classes(), 
    save_every = 1,
    loss_function=torch.nn.functional.cross_entropy, 
    early_stopping = 10
)
train_dict = trainer.fit(model, train_loader, val_loader)
losses, cms = train_dict["losses"], train_dict["cms"]
if "checkpoint" in train_dict:
    state_dict = torch.load(os.path.join(train_dict["checkpoint"], "model.pt"), map_location="cpu")
    model.load_state_dict(state_dict)

for t_ds, name in zip(test_ds, test_ds_names):
    test_loader = DataLoader(t_ds, batch_size=batch_size, shuffle=False, sampler=None, num_workers=num_workers_dataloader, collate_fn=partial(batch_collate, ignore_list=["time", "patient", "data"]), drop_last=False, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    logger.context(f"{name}")
    for r in [1,2,3,4,5,10]:
        logger.context(f"r={r}")

        trainer.n_repeat_test = r
        test_loss, test_cm = trainer.test(model, test_loader)

        record = {
            "repeat":r,
            "test_loss":test_loss,
            "test_cm":test_cm,
            "train_loss":losses,
            "train_cm":cms,
            "dataset":name,
            "classes":train_multi_ds.get_classes(),
        }
        if "best_model" in train_dict:
            record["best_model"] = train_dict["best_model"]

        append_to_jsonl(experiment_name, record)
        logger.uncontext()
    logger.uncontext()

logger.end_run()