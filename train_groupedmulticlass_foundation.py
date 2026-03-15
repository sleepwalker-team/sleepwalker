#!/bin/env python3

import argparse
import inspect
import os

import mlflow

from sleepwalker.datasets.augmentation.RandomPolarityFlip import RandomPolarityFlip
from sleepwalker.datasets.augmentation.RandomResampleJitter import RandomResampleJitter

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from functools import partial
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.multiprocessing as mp

from sleepwalker.datasets import ChannelConfig
from sleepwalker.datasets.ABC import ABC
from sleepwalker.datasets.Apples import Apples
from sleepwalker.datasets.Basedataset import BaseDataset, batch_collate
from sleepwalker.datasets.CAP import CAP
from sleepwalker.datasets.ISRUC import ISRUC
from sleepwalker.datasets.MNC import MNC
from sleepwalker.datasets.MROS import MROS
from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.NCHSDB import NCHSDB
from sleepwalker.datasets.SHHS import SHHS
from sleepwalker.datasets.SVUH_UCD import SVUH_UCD
from sleepwalker.datasets.SleepEDFx import SleepEDFx
from sleepwalker.datasets.augmentation.TimeShiftAndCrop import TimeShiftAndCrop
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.utils import estimate_class_cnts, get_edf_files_in_repo, random_split
from sleepwalker.datasets import Ruhrlandklinik
from sleepwalker.models import SleepTransformer
from sleepwalker.models.AttnSleep import AttnSleep
from sleepwalker.models.MRASleepNet import MRASleepNet
from sleepwalker.models.SeqSleepNet import SeqSleepNet
from sleepwalker.models.TinySleepNet import TinySleepNet
from sleepwalker.models.USleep import USleep
from sleepwalker.models.preprocessors.Crop import Crop
from sleepwalker.trainer.GroupedChannelMulticlassTrainer import GroupedChannelMulticlassTrainer
from sleepwalker.trainer.losses import class_weights_for_loss, dice_loss
from sleepwalker.trainer.utils import append_to_jsonl
from sleepwalker.utils import logger, MlflowSink, suppress_stdout_logging

from lamarr_energy_tracker import EnergyTracker

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

mp.set_sharing_strategy('file_system')

def build_dataset(edf_path: str|os.PathLike, clazz, event_mapping, fs: float, total_input:str, target_resolution: str, num_workers_dataset: int, channels: list[str], get_item_fn, test_frac: Optional[float] = 0.1, transform = None, dry_run:bool = False, rereference:Optional[List[List[str]]] = None):
    channel_cfg = [ChannelConfig(name=c, normalizer=EEGFilterNormalizer(fs = fs, lowcut=0.5, notch_freq=None)) for c in channels] 
    
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    if dry_run:
        edf_files = edf_files[:2]
        test_frac = 0.5

    if not test_frac or test_frac <= 0:
        with suppress_stdout_logging(logger):
            train_ds = clazz(
                channels = channel_cfg,
                patients = edf_files,
                num_workers = num_workers_dataset,
                sample_frequency = fs,
                event_mapping = event_mapping,
                get_item = get_item_fn,
                total_input = total_input, 
                target_resolution = target_resolution,
                transform = transform,
                rereference = rereference
            )
        test_ds = None
        logger.info(f"Loaded {train_ds.get_n_patients()} / {len(edf_files)} patients.")
    else:
        train_patients, test_patients = random_split(edf_files, test_frac=test_frac)
        with suppress_stdout_logging(logger):
            logger.context("TRAIN")
            train_ds = clazz(
                channels = channel_cfg,
                patients = train_patients,
                num_workers = num_workers_dataset,
                sample_frequency = fs,
                event_mapping = event_mapping,
                get_item = get_item_fn,
                total_input = total_input, 
                target_resolution = target_resolution,
                transform = transform,
                rereference = rereference
            )
            logger.uncontext()
            logger.context("TEST")
            test_ds = clazz(
                    channels = channel_cfg,
                    patients = test_patients,
                    num_workers = num_workers_dataset,
                    sample_frequency = fs,
                    event_mapping = event_mapping,
                    get_item = get_item_fn,
                    total_input = total_input, 
                    target_resolution = target_resolution,
                    transform = transform,
                    rereference = rereference
                )
            logger.uncontext()
    
        logger.info(f"Loaded {train_ds.get_n_patients()} / {len(train_patients)} patients for training and {test_ds.get_n_patients()} / {len(test_patients)} patients for validation")
    return train_ds, test_ds

folder = "/raid/"
dataset_cfg = {
    "abc": {
        "clazz": ABC, 
        "edf_path": f"/{folder}/sleepwalker/abc",
        "event_mapping": {
            "wake|0": "wake",
            "stage 1 sleep|1": "n1",
            "stage 2 sleep|2": "n2",
            "stage 3 sleep|3": "n3",
            "rem sleep|5": "rem"
        },
        "channels": ["F3", "F4", "C3", "C4", "O1", "O2", "M1", "M2"],
        "rereference": [["F3", "F4", "C3", "C4", "O1", "O2", "M1", "M2"]],
    },
    "cap": { 
        "clazz": CAP, 
        "edf_path": f"/{folder}/sleepwalker/cap",
        "event_mapping": {
            "S1":"n1",
            "S2":"n2",
            "S3":"n3",
            "S4":"n3",
            "R":"rem",
            "W":"wake"
        },
        "channels": ["F4-C4", "P4-O2", "C4-P4", "C4-A1"],
        "rereference": [["F4-C4", "P4-O2", "C4-P4", "C4-A1"]]
    },
    "isruc": { 
        "clazz": ISRUC, 
        "edf_path": f"/{folder}/sleepwalker/isruc",
        "event_mapping": {
            "N1":"n1",
            "N2":"n2",
            "n2":"n2",
            "N3":"n3",
            "R":"rem",
            "W":"wake",
            "w":"wake"
        },
        "channels": ["F3-M2", "C3-M2", "O1-M2", "F4-M1", "C4-M1", "O2-M1"],
        "rereference": [["F3-M2", "C3-M2", "O1-M2", "F4-M1", "C4-M1", "O2-M1"]],
    },
    "mnc": { 
        "clazz": MNC, 
        "edf_path": f"/{folder}/sleepwalker/mnc/cnc",
        "event_mapping": {
            "nrem1":"n1",
            "nrem2":"n2",
            "nrem3":"n3",
            "rem":"rem",
            "wake":"wake",
        },
        "channels": ["F3", "F4", "C4", "C3"],
        "rereference": [["F3", "F4", "C4", "C3"]]
    },
    "nchsdb": { 
        "clazz": NCHSDB, 
        "edf_path": f"/{folder}/sleepwalker/nchsdb/sleep_data",
        "event_mapping": {
            "Sleep stage 1":"n1",
            "Sleep stage 2":"n2",
            "Sleep stage 3":"n3",
            "Sleep stage R":"rem",
            "Sleep stage W":"wake",
            "Sleep stage N1": "n1",
            "Sleep stage N2": "n2",
            "Sleep stage N3": "n3"
        },
        "channels": ["EEG C3-M2", "EEG O2-M1", "EEG O1-M2", "EEG F3-M2", "EEG C4-M1", "EEG F4-M1"],
        "rereference": [["EEG C3-M2", "EEG O2-M1", "EEG O1-M2", "EEG F3-M2", "EEG C4-M1", "EEG F4-M1"]]
    }, 
    "svuhucd": { 
        "clazz": SVUH_UCD, 
        "edf_path": f"/{folder}/sleepwalker/svuh-ucd",
        "event_mapping": {
            "0": "wake",
            "1": "rem",
            "2": "n1",
            "3": "n2",
            "4": "n3",
            "5": "n3"
        },
        "channels": ["C3A2", "C4A1"],
        "rereference": [["C3A2", "C4A1"]]
    },
    "shhs": { 
        "clazz": SHHS, 
        "edf_path": f"/{folder}/sleepwalker/shhs",
        "event_mapping": {
            "wake|0": "wake",
            "stage 1 sleep|1": "n1",
            "stage 2 sleep|2": "n2",
            "stage 3 sleep|3": "n3",
            "stage 4 sleep|4": "n3",
            "rem sleep|5": "rem"
        },
        "channels": ["EEG"]
    },  
    # TEST
    "apples": {
        "clazz": Apples, 
        "edf_path": f"/{folder}/sleepwalker/apples/polysomnography",
        "event_mapping": {
            "N1":"n1",
            "N2":"n2",
            "N3":"n3",
            "R":"rem",
            "W":"wake"
        },
        "channels": ["C3_M2", "C4_M1", "O2_M1", "O1_M2"],
        "rereference": [["C3_M2", "C4_M1", "O2_M1", "O1_M2"]]
    }, 
    "mros": {
        "clazz": MROS, 
        "edf_path": f"/{folder}/sleepwalker/mros",
        "event_mapping": {
            "wake|0": "wake",
            "stage 1 sleep|1": "n1",
            "stage 2 sleep|2": "n2",
            "stage 3 sleep|3": "n3",
            "stage 4 sleep|4": "n3",
            "rem sleep|5": "rem"
        },
        "channels": ["C3", "C4"],
        "rereference": [["C3", "C4"]]
    },
    "ruhrland": { 
        "clazz": Ruhrlandklinik, 
        "edf_path": f"/{folder}/sleepwalker/ruhrlandklinik/raw",
        "event_mapping": {
            "wach": "wake",
            "n1": "n1",
            "n2": "n2",
            "n3": "n3",
            "rem": "rem"
        },
        "channels": ["F3", "F4", "C3", "C4", "O1", "O2", "M1", "M2"],
        "rereference": [["F3", "F4", "C3", "C4", "O1", "O2", "M1", "M2"]]
    },
    "sleepedfx": {
        "clazz": SleepEDFx, 
        "edf_path": f"/{folder}/sleepwalker/sleep-edfx",
        "event_mapping": {
            "sleep stage w": "wake",
            "sleep stage 1": "n1",
            "sleep stage 2": "n2",
            "sleep stage 3": "n3",
            "sleep stage 4": "n3",
            "sleep stage r": "rem"
        },
        "channels": ["EEG Fpz-Cz", "EEG Pz-Oz"],
        "rereference": [["EEG Fpz-Cz", "EEG Pz-Oz"]]
    },
}

model_cfg = {
    "sleeptransformer":{
        "clazz": SleepTransformer,
        "total_input": "630s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=50),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        "loss_function":torch.nn.functional.cross_entropy
    },
    "sleeptransformer-aug":{
        "clazz": SleepTransformer,
        "total_input": "630s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=50),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        "loss_function":torch.nn.functional.cross_entropy,
        "transform": [
            RandomPolarityFlip(p=0.3, flip_p=0.1),
            RandomResampleJitter(p=0.5, scale=0.05),
        ]
    },
    "attnsleep":{
        "clazz": AttnSleep,
        "total_input": "30s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=50),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3, amsgrad=True),
        "loss_function":torch.nn.functional.cross_entropy,
        "loss_mode": "inverse-log", 
        # I am not sure if these class_weights are correct. They are not explicitly mentioned in the paper and only available in the 
        # original source code. The source code first maps classes to indices and then merges the indicies to combine S3/S4 into N3. 
        # Finally, scores are assigned based on the indicies. I did not execute / debug the code, so I am not 100% sure if these weights
        # match.
        "class_weights": {"wake": 1.5, "n1": 2, "n2": 1.5, "n3": 1, "rem": 1.5}
    },
    "mrasleepnet":{
        "clazz": MRASleepNet,
        "total_input": "40s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=50),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=1e-4),
        "loss_function":torch.nn.functional.cross_entropy,
        "loss_mode": "inverse-log", 
        # I am not sure if these class_weights are correct. They are not explicitly mentioned in the paper and only available in the 
        # original source code. The source code first maps classes to indices and then merges the indicies to combine S3/S4 into N3. 
        # Finally, scores are assigned based on the indicies. I did not execute / debug the code, so I am not 100% sure if these weights
        # match.
        "class_weights": {"wake": 1.5, "n1": 2, "n2": 1.5, "n3": 1, "rem": 1.5}
    },
    "seqsleepnet":{
        "clazz": SeqSleepNet,
        "total_input": "900s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=50),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=1e-4),
        "loss_function":torch.nn.functional.cross_entropy
    },
    "tinysleepnet":{
        "clazz": TinySleepNet,
        "total_input": "500s", # We use data augmentation that shifts the input slightly across the time axis. Ultimately, we want to have an input of 450s (chunk_len (=15) * 30s), which we get by cropping the timeseries (see preprocessor below). To correct augment the data, we require a slightly larger input which we get here
        "total_input_model": "450s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=50),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1.0e-3),
        "loss_function":torch.nn.functional.cross_entropy,
        "loss_mode": "regular", 
        "class_weights": {"wake": 1, "n1": 1.5, "n2": 1, "n3": 1, "rem": 1},
        "transform": [TimeShiftAndCrop(max_shift="60s", fill="closest", sampling_frequency=100, output_size="450s")],
        "seq_len":15,
        "use_lstm":True
    },
    "usleep": { 
        "clazz": USleep,
        "total_input": "1050s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=50),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=5e-6, amsgrad=True),
        "loss_function":dice_loss,
        "loss_mode": "regular", 
        "channel": [32, 64, 128, 256],
        "maxpool": [10, 8, 6, 4,],
        "kernel": [5, 5, 5, 5,],
        "activation": "relu",
        "norm": "batch", 
        "mlp_size": 128,
        "balance_batches":True
    }
}

def create_model(cls, **kwargs):
    # only pass what the class supports
    sig = inspect.signature(cls)
    params = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**params)

def run(model_name, train_datasets, test_datasets, dry_run):
    if model_name not in model_cfg:
        logger.warning(f"Did not find {model_name} in model_cfg -- Ignoring this model")
        return

    # common parameter
    batch_size = 128
    epochs = 100 if not dry_run else 1
    target_resolution = "30s"
    n_samples = 250_000 if not dry_run else 1_000 
    experiment_name = "grouped-multiclass-foundation-rereferenced"
    num_workers_dataset = 16
    num_workers_dataloader = 16
    sample_frequency = 100

    # model specific parameters
    total_input = model_cfg[model_name]["total_input"]
    model_clazz = model_cfg[model_name]["clazz"]
    lr_scheduler = model_cfg[model_name]["lr_scheduler"]
    optimizer = model_cfg[model_name]["optimizer"]
    loss_function = model_cfg[model_name]["loss_function"]
    loss_mode = model_cfg[model_name].get("loss_mode", "regular")
    class_weights = model_cfg[model_name].get("class_weights", {})
    transform = model_cfg[model_name].get("transform", None)
    balance_batches = model_cfg[model_name].get("balance_batches", False)

    all_channels = set([c for ds in train_datasets for c in dataset_cfg[ds]["channels"]])
    all_channels = all_channels.union(set([c for ds in test_datasets for c in dataset_cfg[ds]["channels"]]))
    groups = [list(all_channels)] # Below code supports multiple groups so that we can sample fro multiple channels. Technically, we only want to channel from the EEG group for 1 target channel, hence a list of list with a single entry

    if dry_run:
        logger.info("Performing dry run to test pipeline! Each dataset loads at most 2 patients")
        experiment_name += "-dev"
    else:
        tracker = EnergyTracker(project_name=experiment_name)
        tracker.start()
    
    # with open(os.path.expanduser("~/mlflow/auth_config.ini"),"r") as f:
    #     TRACKING_URI=f.read().strip()
    # ARTIFACT_URI=f"/raid/mlruns"

    # mlflow_sink = MlflowSink(tracking_uri=TRACKING_URI, experiment=experiment_name, artifact_uri=ARTIFACT_URI)
    # logger.add_sink(mlflow_sink)
    logger.start_run(run_name=experiment_name,tags={"model":model_name})
    
    train_ds = []
    val_ds = []

    for ds in train_datasets:
        if ds not in dataset_cfg:
            logger.warning(f"Did not find {ds} in dataset_cfg -- Ignoring this dataset")
            continue
        
        logger.context(ds)
        train, val = build_dataset(**dataset_cfg[ds], 
            fs = sample_frequency, 
            total_input = total_input, 
            target_resolution = target_resolution, 
            get_item_fn=partial(GroupedChannelMulticlassTrainer.get_item, groups=groups),
            num_workers_dataset = num_workers_dataset,
            test_frac=0.3,
            transform=transform, 
            dry_run=dry_run
        )
        logger.uncontext()
        train_ds.append(train)
        val_ds.append(val)

    test_ds = []
    for ds in test_datasets:
        if ds not in dataset_cfg:
            logger.warning(f"Did not find {ds} in dataset_cfg -- Ignoring this dataset")
            continue
        logger.context(ds)
        test, _ = build_dataset(**dataset_cfg[ds], 
            fs = sample_frequency, 
            # TinySleepNet uses augmentation that changes the model input size. For testing we have to be consistent
            total_input = model_cfg[model_name]["total_input_model"] if "total_input_model" in model_cfg[model_name] else total_input, 
            target_resolution = target_resolution, 
            get_item_fn=partial(GroupedChannelMulticlassTrainer.get_item, groups=groups),
            num_workers_dataset = num_workers_dataset,
            test_frac=0, 
            dry_run=dry_run
        )
        logger.uncontext()
        test_ds.append(test)

    if os.path.exists("sleepwalker.log"):
        os.remove("sleepwalker.log")

    train_multi_ds = MultiDataset(train_ds)
    logger.info(f"Loaded {train_multi_ds.get_n_patients()} for training")

    val_multi_ds = MultiDataset(val_ds)
    logger.info(f"Loaded {val_multi_ds.get_n_patients()} for validation")

    test_multi_ds = MultiDataset(test_ds)
    logger.info(f"Loaded {test_multi_ds.get_n_patients()} for testing")

    if "total_input_model" in model_cfg[model_name]:
        # TinySleepNet uses augmentation that changes the model input size. Hence its actual input size is different from total_input 
        freq = pd.to_timedelta(1.0 / sample_frequency, unit="s")
        total_input = pd.to_timedelta(model_cfg[model_name]["total_input_model"])
        ts_len = int(total_input.total_seconds() / freq.total_seconds())
    else:
        ts_len = train_multi_ds.get_timeseries_len()

    n_channels = 1
    model = create_model(cls=model_clazz, classes=train_multi_ds.get_classes(), ts_len=ts_len, n_channels=n_channels, sampling_frequency=sample_frequency, **model_cfg[model_name])
    logger.info(f"Input data is {ts_len} x {n_channels}")
    model_stats = summary(model, input_size=(1, ts_len, n_channels), depth=5, row_settings=["hide_recursive_layers"])

    if balance_batches:
        class_cnts = estimate_class_cnts(train_multi_ds, n_samples, num_workers_dataloader, batch_size)
        class_cnts_list = [class_cnts.get(c, 1) for c in train_multi_ds.get_classes()]
        for i in range(len(train_datasets)):
            train_multi_ds.datasets[i].get_item_callback = partial(GroupedChannelMulticlassTrainer.get_item, groups=groups, class_cnts=class_cnts_list)
    else:
        class_cnts = None

    if loss_mode != "regular":
        if class_cnts is None: 
            class_cnts = estimate_class_cnts(train_multi_ds, n_samples, num_workers_dataloader, batch_size) 
        class_weights = class_weights_for_loss(class_weights, class_cnts, loss_mode)
    
    if len(class_weights) > 0:
        weights_torch = torch.tensor([class_weights.get(c, 1) for c in train_multi_ds.get_classes()], device="cuda")
    else:
        weights_torch = None

    trainer = GroupedChannelMulticlassTrainer(
        groups=groups,
        epochs=epochs, 
        optimizer = optimizer,
        lr_scheduler = lr_scheduler,
        classes = train_multi_ds.get_classes(), 
        save_every = 1,
        loss_function=partial(loss_function, weight=weights_torch), 
        early_stopping = 10
    )

    train_sampler = RandomSampler(train_multi_ds, num_samples = n_samples) if n_samples is not None else None
    train_loader = DataLoader(train_multi_ds, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=num_workers_dataloader, collate_fn=partial(batch_collate, ignore_list=["time", "patient", "data"]), drop_last=False, persistent_workers=True, prefetch_factor=2, pin_memory=True) ### prefetch_factor=12

    val_sampler = RandomSampler(val_multi_ds, num_samples = n_samples) if n_samples is not None else None
    val_loader = DataLoader(val_multi_ds, batch_size=batch_size, shuffle=val_sampler is None, sampler=val_sampler, num_workers=num_workers_dataloader, collate_fn=partial(batch_collate, ignore_list=["time", "patient", "data"]), drop_last=False, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    train_dict = trainer.fit(model, train_loader, val_loader)
    losses, cms = train_dict["losses"], train_dict["cms"]
    if "checkpoint" in train_dict:
        with torch.inference_mode():
            state_dict = torch.load(os.path.join(train_dict["checkpoint"], "model.pt"), map_location="cpu")
            model.load_state_dict(state_dict)

    for t_ds, name in zip(test_ds, test_datasets):
        test_loader = DataLoader(t_ds, batch_size=batch_size, shuffle=False, sampler=None, num_workers=num_workers_dataloader, collate_fn=partial(batch_collate, ignore_list=["time", "patient", "data"]), drop_last=False, persistent_workers=True, prefetch_factor=2, pin_memory=True)

        logger.context(f"{name}")
        for r in [1,2,3,4,5,10]:
            logger.context(f"r={r}")

            trainer.n_repeat_test = r
            test_loss, test_cm = trainer.test(model, test_loader)

            record = {
                "model": model_name,
                "repeat":r,
                "test_loss":test_loss,
                "test_cm":test_cm,
                "train_loss":losses,
                "train_cm":cms,
                "dataset":name,
                "classes":train_multi_ds.get_classes(),
                "artifact_root": train_dict["checkpoint"] if "checkpoint" in train_dict else "",
            }
            if "best_model" in train_dict:
                record["best_model"] = train_dict["best_model"]

            append_to_jsonl(f"{experiment_name}_{model_name}", record)
            logger.uncontext()
        logger.uncontext()

    if not dry_run: tracker.stop()
    logger.end_run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a sleep-staging model on multiple datasets with grouped EEG channels.')
    parser.add_argument("--model", help='The model to be trained. Appropriate training parameters are choosen automatically', required=False, default="sleeptransformer", type=str)
    parser.add_argument("--train", help='Datasets to train the model on', required=False, default=["abc", "cap", "isruc", "mnc", "nchsdb", "svuhucd", "shhs"], type=str, nargs="+")
    parser.add_argument("--test", help='Datasets to test the model on', required=False, default=["apples", "mros", "ruhrland", "sleepedfx"], type=str, nargs="+")
    parser.add_argument("--dry", help='If set, performs a quick test run without actually training the model', action="store_true")
    args = parser.parse_args()
    run(args.model, args.train, args.test, args.dry)