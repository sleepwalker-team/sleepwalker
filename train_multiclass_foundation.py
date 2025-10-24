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
from sleepwalker.datasets.SHHS import SHHS
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.utils import get_edf_files_in_repo, kfold_split, random_split
from sleepwalker.datasets import Ruhrlandklinik

from sleepwalker.models import SleepTransformer
from sleepwalker.trainer.GroupedChanelMulticlassTrainer import GroupedChanelMulticlassTrainer
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.utils import append_to_jsonl
from sleepwalker.utils import logger, MlflowSink

#batch_size = 512
batch_size = 64
epochs = 100
total_input = "630s"
target_resolution = "30s"
n_samples = 500_000
tracking_uri = "file:/raid/mlflow/"
experiment_name = "foundation"
num_workers_dataset = 32

logger.add_sink(MlflowSink(tracking_uri="sqlite:///mlflow.sqlite", experiment=experiment_name))

def build_abc(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    train_patients, test_patients = random_split(edf_files, test_frac=0.25)
    
    def _create(patients):
        channels = [
            ChannelConfig(name="C4", normalizer=EEGFilterNormalizer()),
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
            sample_frequency = 100,
            event_mapping = event_mapping,
            get_item = MulticlassTrainer.get_item
            online_filtering = True, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    return _create(train_patients), _create(test_patients)

def build_apples(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    train_patients, test_patients = random_split(edf_files, test_frac=0.25)
    
    def _create(patients):
        channels = [
            ChannelConfig(name="C4_M1", normalizer=EEGFilterNormalizer()),
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
            sample_frequency = 100,
            event_mapping = event_mapping,
            get_item = MulticlassTrainer.get_item
            online_filtering = True, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    return _create(train_patients), _create(test_patients)

def build_cap(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    edf_files = edf_files[:10] #XXX

    train_patients, test_patients = random_split(edf_files, test_frac=0.25)
    
    def _create(patients):
        channels = [
            # ChannelConfig(name="F4-C4", normalizer=EEGFilterNormalizer()),
            ChannelConfig(name="C4-P4", normalizer=EEGFilterNormalizer()),
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
            sample_frequency = 100,
            event_mapping = event_mapping,
            get_item = MulticlassTrainer.get_item
            online_filtering = True, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    return _create(train_patients), _create(test_patients)

def build_isruc(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    train_patients, test_patients = random_split(edf_files, test_frac=0.25)
    
    def _create(patients):
        channels = [
            ChannelConfig(name="C4-M1", normalizer=EEGFilterNormalizer()),
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
            sample_frequency = 100,
            event_mapping = event_mapping,
            get_item = MulticlassTrainer.get_item
            online_filtering = True, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    return _create(train_patients), _create(test_patients)

def build_mnc(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    train_patients, test_patients = random_split(edf_files, test_frac=0.25)
    
    def _create(patients):
        channels = [
            ChannelConfig(name="C4", normalizer=EEGFilterNormalizer()),
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
            sample_frequency = 100,
            event_mapping = event_mapping,
            get_item = MulticlassTrainer.get_item
            online_filtering = True, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    return _create(train_patients), _create(test_patients)

def build_mros(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    train_patients, test_patients = random_split(edf_files, test_frac=0.25)
    
    def _create(patients):
        channels = [
            ChannelConfig(name="C4", normalizer=EEGFilterNormalizer()),
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
            sample_frequency = 100,
            event_mapping = event_mapping,
            get_item = MulticlassTrainer.get_item
            online_filtering = True, 
            total_input = total_input, 
            target_resolution = target_resolution
        )

    return _create(train_patients), _create(test_patients)

def build_ruhrland(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    edf_files = edf_files[:10] #XXX

    dataset = Ruhrlandklinik(
        channels = [
            ChannelConfig(name="C4", normalizer=EEGFilterNormalizer()),
        ],
        return_nox = False,
        patients = edf_files,
        num_workers = num_workers_dataset,
        sample_frequency = 100,
        event_mapping = {   
            "wach": "wake",
            "n1": "n1",
            "n2": "n2",
            "n3": "n3",
            "rem": "rem"
        },
        get_item = MulticlassTrainer.get_item 
        online_filtering = True, 
        total_input = total_input, 
        target_resolution = target_resolution
    )

    return dataset

def build_sleepedfx(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)

    dataset = SleepEDFx(
        channels = [
            ChannelConfig(name="C4", normalizer=EEGFilterNormalizer()),
        ],
        return_nox = False,
        patients = edf_files,
        num_workers = num_workers_dataset,
        sample_frequency = 100,
        event_mapping = {   
            "wach": "wake",
            "n1": "n1",
            "n2": "n2",
            "n3": "n3",
            "rem": "rem"
        },
        get_item = MulticlassTrainer.get_item 
        online_filtering = True, 
        total_input = total_input, 
        target_resolution = target_resolution
    )

    return dataset

def build_shhs(edf_path):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)

    dataset = SHHS(
        channels = [
            ChannelConfig(name="EEG", normalizer=EEGFilterNormalizer()),
        ],
        patients = edf_files,
        num_workers = num_workers_dataset,
        sample_frequency = 100,
        event_mapping = {
            "wake|0": "wake",
            "stage 1 sleep|1": "n1",
            "stage 2 sleep|2": "n2",
            "stage 3 sleep|3": "n3",
            "stage 4 sleep|4": "n3",
            "rem sleep|5": "rem"
        },
        get_item = MulticlassTrainer.get_item 
        online_filtering = True, 
        total_input = total_input, 
        target_resolution = target_resolution
    )

    return dataset

logger.start_run(run_name=f"Grouped multiclass foundation")

ds = []
# logger.context("ABC")
# ds.append(build_abc("/raid/sleepwalker/abc"))
# logger.uncontext()

# logger.context("Apples")
# ds.append(build_apples("/raid/sleepwalker/apples/polysomnography"))
# logger.uncontext()

logger.context("CAP")
ds.append(build_cap("/raid/sleepwalker/cap"))
logger.uncontext()

# logger.context("ISRUC")
# ds.append(build_isruc("/raid/sleepwalker/isruc"))
# logger.uncontext()

# logger.context("MNC")
# ds.append(build_mnc("/raid/sleepwalker/mnc/cnc"))
# logger.uncontext()

# logger.context("MROS")
# ds.append(build_mros("/raid/sleepwalker/mros"))
# logger.uncontext()

train_ds = [d[0] for d in ds]
val_ds = [d[1] for d in ds]

test_ds = []
logger.context("Ruhrland")
test_ds.append(build_ruhrland("/raid/sleepwalker/ruhrlandklinik/raw"))
logger.uncontext()

# logger.context("SHHS")
# test_ds.append(build_shhs("/raid/sleepwalker/shhs"))
# logger.uncontext()

if os.path.exists("sleepwalker.log"):
    os.remove("sleepwalker.log")

train_multi_ds = MultiDataset(train_ds)
logger.info(f"Loaded {train_multi_ds.n_patients()} patients for training")

val_multi_ds = MultiDataset(val_ds)
logger.info(f"Loaded {val_multi_ds.n_patients()} patients for validation")

test_multi_ds = MultiDataset(test_ds)
logger.info(f"Loaded {test_multi_ds.n_patients()} patients for testing")

train_sampler = RandomSampler(train_multi_ds, num_samples = n_samples) if n_samples is not None else None
train_loader = DataLoader(train_multi_ds, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=16, pin_memory=True, collate_fn=partial(batch_collate, ignore_list=["time", "patient", "data"]), drop_last=False, persistent_workers=False)

val_sampler = RandomSampler(val_multi_ds, num_samples = n_samples) if n_samples is not None else None
val_loader = DataLoader(val_multi_ds, batch_size=batch_size, shuffle=val_sampler is None, sampler=val_sampler, num_workers=16, pin_memory=True, collate_fn=partial(batch_collate, ignore_list=["time", "patient", "data"]), drop_last=False, persistent_workers=False)

test_loader = DataLoader(test_multi_ds, batch_size=batch_size, shuffle=True, sampler=None, num_workers=16, pin_memory=True, collate_fn=partial(batch_collate, ignore_list=["time", "patient", "data"]), drop_last=False, persistent_workers=False)

model = SleepTransformer(classes=train_multi_ds.get_classes(), n_channels=len(groups))
model_stats = summary(model, input_size=(1, train_multi_ds.get_timeseries_len(), len(groups)), depth=5, row_settings=["hide_recursive_layers"])

trainer = MulticlassTrainer(
    epochs=epochs, 
    optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
    lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=50),
    classes = train_multi_ds.get_classes(), 
    save_every = 10,
    loss_function=torch.nn.functional.cross_entropy, 
)
losses, cms = trainer.fit(model, train_loader)

# TODO store final model
for r in [1,2,3,4,5,10]:
    logger.context(f"r={r}")
    test_loss, test_cm = trainer.test(model, test_loader, n_apply_repeats=r)

    record = {
        "repeat":r,
        "test_loss":test_loss,
        "test_cm":test_cm,
        "train_loss":losses,
        "train_cm":cms,
        "classes":train_multi_ds.get_classes(),
    }
    append_to_jsonl(experiment_name, record)
    logger.uncontext()

logger.end_run()