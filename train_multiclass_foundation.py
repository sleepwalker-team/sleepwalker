#!/bin/env python3

import logging
import os
import torch
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader

from torchinfo import summary
from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets import ChannelConfig
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.utils import get_edf_files_in_repo, kfold_split, random_split
from sleepwalker.datasets import Ruhrlandklinik

from sleepwalker.models import SleepTransformer
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.utils import append_to_jsonl
from sleepwalker.utils import logger, MlflowSink

total_input = "630s"
target_resolution = "30s"

def build_abc(edf_path, ending="edf"):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True, ending=ending)
    train_patients, test_patients = random_split(edf_files, test_frac=0.25)
    
    train_dataset = ABC(
        annotator="nsrr",
        channels = [
            ChannelConfig(name="C4", normalizer=None),
        ],
        patients = train_patients,
        num_workers = 8,
        sample_frequency = 100,
        event_mapping = {
            "wake|0": "wake",
            "stage 1 sleep|1": "n1",
            "stage 2 sleep|2": "n2",
            "stage 3 sleep|3": "n3",
            "rem sleep|5": "rem"
        },
        get_item = MulticlassTrainer.get_item, # Implements 50% rule for multi-class classification
        online_filtering = True, # Let the dataset reject data points that are not usable for training by calling MulticlassTrainer.get_item
        total_input = total_input, 
        target_resolution = target_resolution
    )

# Parameters for this run
edf_folder = "/raid/data/ruhrlandklinik/raw/train-test-2023"
mode = "xval"                      
n_splits = 5
batch_size = 128
epochs = 100

n_samples = None #10_000
tracking_uri = "file:./mlruns"
experiment_name = "ruhrland_sleeptransformer"

# Function that is called by the Basedataset to filter out patients _before_ loading them 
# If true is returned => try to load patient
# If false is returned => try not to load patient
def filter_patient(edf_path):
    meta = read_edf_meta(edf_path)
    return "C4-M1" in meta["signals"]

def build_loader(patients):
    dataset = Ruhrlandklinik(
        channels = [ChannelConfig(name="C4-M1", normalizer=None)],
        patients = patients,
        num_workers = 8,
        sample_frequency = 100,
        event_mapping = {
            "wach": "wake",
            "n1": "n1",
            "n2": "n2",
            "n3": "n3",
            "rem": "rem"
        },
        get_item = MulticlassTrainer.get_item, # Implements 50% rule for multi-class classification
        online_filtering = True, # Let the dataset reject data points that are not usable for training by calling MulticlassTrainer.get_item
        total_input = total_input, 
        target_resolution = target_resolution
    )

    sampler = RandomSampler(dataset, num_samples = n_samples) if n_samples is not None else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=8, pin_memory=True, collate_fn=batch_collate, drop_last=False, persistent_workers=True)

    return loader, dataset






all_patients = get_edf_files_in_repo(edf_folder, recursive=False)
all_patients = [p for p in all_patients if filter_patient(p)]

# Logging is now vastly simplified: 
#   We have a global singleton logger (from sleepwalker.utils import logger) that can be used for {metric,artifacts,text,...} logging
#   We can add sinks for logging to different backends (file, mlflow, etc). Per default we log to file and TQDM/Console
if os.path.exists("sleepwalker.log"):
    # Reset log file, per default we always append to the current file
    os.remove("sleepwalker.log")

# Add an mlflow sink with the appropriate experiment name and backend URI
logger.add_sink(MlflowSink(tracking_uri="sqlite:///mlflow.sqlite", experiment="MyExperiment")) # can also be file:... as backend

for i, (train_patients, test_patients) in enumerate(kfold_split(all_patients, n_splits=n_splits)):
    logger.start_run(run_name=f"XVAL {i}") # Set the run_name for this experiment

    train_loader, dataset = build_loader(train_patients)

    model = SleepTransformer(classes=dataset.get_classes(), n_channels=1)
    model_stats = summary(model, input_size=(1, dataset.get_timeseries_len(), 1), depth=5, row_settings=["hide_recursive_layers"])

    trainer = MulticlassTrainer(
        epochs=epochs, 
        optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=50),
        classes = dataset.get_classes(), 
        save_every = 10,
        loss_function=torch.nn.functional.cross_entropy, 
    )
    losses, cms = trainer.fit(model, train_loader)

    test_loader, _ = build_loader(test_patients)
    test_loss, test_cm = trainer.test(model, test_loader)

    record = {
        "xval":i,
        "test_loss":test_loss,
        "test_cm":test_cm,
        "train_loss":losses,
        "train_cm":cms,
        "classes":dataset.get_classes(),
    }
    append_to_jsonl(experiment_name, record)

