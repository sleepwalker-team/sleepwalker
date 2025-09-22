#!/bin/env python3

import json
import os
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader

from torchinfo import summary
from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets import ChannelConfig
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.utils import get_edf_files_in_repo, kfold_split
from sleepwalker.datasets import Ruhrlandklinik

from sleepwalker.models import SleepTransformer
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
# from sleepwalker.trainer.logger.MLFlowLogger import MlflowLogger
from sleepwalker.trainer.utils import append_to_jsonl
from sleepwalker.utils import logger


edf_folder = "/raid/data/ruhrlandklinik/raw/train-test-2023"
mode = "xval"                      
n_splits = 5
batch_size = 128
epochs = 100
total_input = "630s"
target_resolution = "30s"
n_samples = None #10_000
tracking_uri = "file:./mlruns"
experiment_name = "ruhrland_sleeptransformer"

def filter_patient(edf_path):
    meta = read_edf_meta(edf_path)
    return "C4-M1" in meta["signals"]

def build_loader(patients, prefix=""):
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
        filter_patient = filter_patient,
        get_item = MulticlassTrainer.get_item,
        online_filtering = True,
        total_input = total_input, 
        target_resolution = target_resolution
    )

    sampler = RandomSampler(dataset, num_samples = n_samples) if n_samples is not None else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=8, pin_memory=True, collate_fn=batch_collate, drop_last=False, persistent_workers=True)

    return loader, dataset

all_patients = get_edf_files_in_repo(edf_folder, recursive=False)

for i, (train_patients, test_patients) in enumerate(kfold_split(all_patients, n_splits=n_splits)):
    logger.context(f"XVAL {i}/{n_splits}")
    #mlflow_logger = MlflowLogger(experiment_name="ruhrland_sleeptransformer", run_name = f"{i}", tracking_uri=tracking_uri)
    mlflow_logger = None

    train_loader, dataset = build_loader(train_patients)

    model = SleepTransformer(classes=dataset.get_classes(), n_channels=1)
    model_stats = summary(model, input_size=(1, dataset.get_timeseries_len(), 1), depth=5, row_settings=["hide_recursive_layers"])

    trainer = MulticlassTrainer(
        epochs=epochs, 
        optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=50),
        classes = dataset.get_classes(), 
        # logger = mlflow_logger,
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
    
    print("")

