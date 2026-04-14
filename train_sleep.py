#!/bin/env python3

from __future__ import annotations

import argparse
import copy
import inspect
import os
from functools import partial

import pandas as pd
import torch
import torch.multiprocessing as mp

# os.environ["OMP_NUM_THREADS"] = "2"
# os.environ["MKL_NUM_THREADS"] = "2"
# os.environ["OPENBLAS_NUM_THREADS"] = "2"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
# os.environ["NUMEXPR_NUM_THREADS"] = "2"

from sleepwalker.datasets import ChannelConfig
from sleepwalker.datasets import Ruhrlandklinik
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.ABC import ABC
from sleepwalker.datasets.Apples import Apples
from sleepwalker.datasets.CAP import CAP
from sleepwalker.datasets.ISRUC import ISRUC
from sleepwalker.datasets.MNC import MNC
from sleepwalker.datasets.MROS import MROS
from sleepwalker.datasets.NCHSDB import NCHSDB
from sleepwalker.datasets.SHHS import SHHS
from sleepwalker.datasets.SVUH_UCD import SVUH_UCD
from sleepwalker.datasets.SleepEDFx import SleepEDFx
from sleepwalker.datasets.augmentation.RandomPolarityFlip import RandomPolarityFlip
from sleepwalker.datasets.augmentation.RandomResampleJitter import RandomResampleJitter
from sleepwalker.datasets.augmentation.TimeShiftAndCrop import TimeShiftAndCrop
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.utils import get_edf_files_in_repo, random_split
from sleepwalker.models import SleepTransformer
from sleepwalker.models.AttnSleep import AttnSleep
from sleepwalker.models.MRASleepNet import MRASleepNet
from sleepwalker.models.SeqSleepNet import SeqSleepNet
from sleepwalker.models.TinySleepNet import TinySleepNet
from sleepwalker.models.USleep import USleep
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.NegativeGroupedChanelMulticlassTrainer import GradReverseTrainer
from sleepwalker.trainer.losses import dice_loss
from sleepwalker.trainer.utils.filtering import filter_patients_by_sleep_time, trim_wake
from sleepwalker.trainer.utils.splits import combine_datasets, split_patients_train_val_test
from sleepwalker.trainer.utils.targets import prepare_multiclass_target
from sleepwalker.utils import logger, suppress_stdout_logging

# torch.set_num_threads(2)
# torch.set_num_interop_threads(1)
mp.set_sharing_strategy("file_system")

TARGET_CLASSES = ["wake", "n1", "n2", "n3", "rem"]
TASK_DEFAULTS = {
    "sample_frequency": 100,
    "target_resolution": "30s",
}

SCORED_SLEEP_LABELS = ["n1", "n2", "n3", "rem"]
BATCH_SIZE = 128
EPOCHS = 100
N_SAMPLES = 250_000
NUM_WORKERS_DATASET = 8
NUM_WORKERS_DATALOADER = 8
TEST_FRAC = 0.3
VAL_FRAC = 0.1
EXPERIMENT_NAME = "multiclass"
DATASET_ROOT = "/raid/sleepwalker"
SLEEP_PERCENTAGE = 0.5
GROUPED_TEST_REPEATS = [1, 2, 3, 4, 5, 10]
SLEEP_TIME_FILTER_QUANTILE = 0.05

DATASET_CFG = {
    "abc": {
        "clazz": ABC,
        "edf_path": "abc",
        "event_mapping": {
            "wake|0": "wake",
            "stage 1 sleep|1": "n1",
            "stage 2 sleep|2": "n2",
            "stage 3 sleep|3": "n3",
            "rem sleep|5": "rem",
        },
        "channels": [ChannelConfig("C4")],
        "grouped_channels": [
            ChannelConfig("F3", group="eeg"),
            ChannelConfig("F4", group="eeg"),
            ChannelConfig("C3", group="eeg"),
            ChannelConfig("C4", group="eeg"),
            ChannelConfig("O1", group="eeg"),
            ChannelConfig("O2", group="eeg"),
            ChannelConfig("M1", group="eeg"),
            ChannelConfig("M2", group="eeg"),
        ],
        "grouped_rereference": [["F3", "F4", "C3", "C4", "O1", "O2", "M1", "M2"]],
    },
    "cap": {
        "clazz": CAP,
        "edf_path": "cap",
        "event_mapping": {
            "S1": "n1",
            "S2": "n2",
            "S3": "n3",
            "S4": "n3",
            "R": "rem",
            "W": "wake",
        },
        "channels": [ChannelConfig("C4-A1")],
        "grouped_channels": [
            ChannelConfig("F4-C4", group="eeg"),
            ChannelConfig("P4-O2", group="eeg"),
            ChannelConfig("C4-P4", group="eeg"),
            ChannelConfig("C4-A1", group="eeg"),
        ],
        "grouped_rereference": [["F4-C4", "P4-O2", "C4-P4", "C4-A1"]],
    },
    "isruc": {
        "clazz": ISRUC,
        "edf_path": "isruc",
        "event_mapping": {
            "N1": "n1",
            "N2": "n2",
            "n2": "n2",
            "N3": "n3",
            "R": "rem",
            "W": "wake",
            "w": "wake",
        },
        "channels": [ChannelConfig("C4-M1")],
        "grouped_channels": [
            ChannelConfig("F3-M2", group="eeg"),
            ChannelConfig("C3-M2", group="eeg"),
            ChannelConfig("O1-M2", group="eeg"),
            ChannelConfig("F4-M1", group="eeg"),
            ChannelConfig("C4-M1", group="eeg"),
            ChannelConfig("O2-M1", group="eeg"),
        ],
        "grouped_rereference": [["F3-M2", "C3-M2", "O1-M2", "F4-M1", "C4-M1", "O2-M1"]],
    },
    "mnc": {
        "clazz": MNC,
        "edf_path": "mnc/cnc",
        "event_mapping": {
            "nrem1": "n1",
            "nrem2": "n2",
            "nrem3": "n3",
            "rem": "rem",
            "wake": "wake",
        },
        "channels": [ChannelConfig("C4")],
        "grouped_channels": [
            ChannelConfig("F3", group="eeg"),
            ChannelConfig("F4", group="eeg"),
            ChannelConfig("C4", group="eeg"),
            ChannelConfig("C3", group="eeg"),
        ],
        "grouped_rereference": [["F3", "F4", "C4", "C3"]],
    },
    "nchsdb": {
        "clazz": NCHSDB,
        "edf_path": "nchsdb/sleep_data",
        "event_mapping": {
            "Sleep stage 1": "n1",
            "Sleep stage 2": "n2",
            "Sleep stage 3": "n3",
            "Sleep stage R": "rem",
            "Sleep stage W": "wake",
            "Sleep stage N1": "n1",
            "Sleep stage N2": "n2",
            "Sleep stage N3": "n3",
        },
        "channels": [ChannelConfig("EEG C4-M1")],
        "grouped_channels": [
            ChannelConfig("EEG C3-M2", group="eeg"),
            ChannelConfig("EEG O2-M1", group="eeg"),
            ChannelConfig("EEG O1-M2", group="eeg"),
            ChannelConfig("EEG F3-M2", group="eeg"),
            ChannelConfig("EEG C4-M1", group="eeg"),
            ChannelConfig("EEG F4-M1", group="eeg"),
        ],
        "grouped_rereference": [["EEG C3-M2", "EEG O2-M1", "EEG O1-M2", "EEG F3-M2", "EEG C4-M1", "EEG F4-M1"]],
    },
    "svuhucd": {
        "clazz": SVUH_UCD,
        "edf_path": "svuh-ucd",
        "event_mapping": {
            "0": "wake",
            "1": "rem",
            "2": "n1",
            "3": "n2",
            "4": "n3",
            "5": "n3",
        },
        "channels": [ChannelConfig("C4A1")],
        "grouped_channels": [ChannelConfig("C3A2", group="eeg"), ChannelConfig("C4A1", group="eeg")],
        "grouped_rereference": [["C3A2", "C4A1"]],
    },
    "shhs": {
        "clazz": SHHS,
        "edf_path": "shhs",
        "event_mapping": {
            "wake|0": "wake",
            "stage 1 sleep|1": "n1",
            "stage 2 sleep|2": "n2",
            "stage 3 sleep|3": "n3",
            "stage 4 sleep|4": "n3",
            "rem sleep|5": "rem",
        },
        "channels": [ChannelConfig("EEG")],
        "grouped_channels": [ChannelConfig("EEG", group="eeg")],
    },
    "apples": {
        "clazz": Apples,
        "edf_path": "apples/polysomnography",
        "event_mapping": {
            "N1": "n1",
            "N2": "n2",
            "N3": "n3",
            "R": "rem",
            "W": "wake",
        },
        "channels": [ChannelConfig("C4_M1")],
        "grouped_channels": [
            ChannelConfig("C3_M2", group="eeg"),
            ChannelConfig("C4_M1", group="eeg"),
            ChannelConfig("O2_M1", group="eeg"),
            ChannelConfig("O1_M2", group="eeg"),
        ],
        "grouped_rereference": [["C3_M2", "C4_M1", "O2_M1", "O1_M2"]],
    },
    "mros": {
        "clazz": MROS,
        "edf_path": "mros",
        "event_mapping": {
            "wake|0": "wake",
            "stage 1 sleep|1": "n1",
            "stage 2 sleep|2": "n2",
            "stage 3 sleep|3": "n3",
            "stage 4 sleep|4": "n3",
            "rem sleep|5": "rem",
        },
        "channels": [ChannelConfig("C4")],
        "grouped_channels": [ChannelConfig("C3", group="eeg"), ChannelConfig("C4", group="eeg")],
        "grouped_rereference": [["C3", "C4"]],
    },
    "ruhrland": {
        "clazz": Ruhrlandklinik,
        "edf_path": "ruhrlandklinik/raw",
        "event_mapping": {
            "wach": "wake",
            "n1": "n1",
            "n2": "n2",
            "n3": "n3",
            "rem": "rem",
        },
        "channels": [ChannelConfig("C4")],
        "grouped_channels": [
            ChannelConfig("F3", group="eeg"),
            ChannelConfig("F4", group="eeg"),
            ChannelConfig("C3", group="eeg"),
            ChannelConfig("C4", group="eeg"),
            ChannelConfig("O1", group="eeg"),
            ChannelConfig("O2", group="eeg"),
            ChannelConfig("M1", group="eeg"),
            ChannelConfig("M2", group="eeg"),
        ],
        "grouped_rereference": [["F3", "F4", "C3", "C4", "O1", "O2", "M1", "M2"]],
    },
    "ruhrland2023": {
        "clazz": Ruhrlandklinik,
        "edf_path": "ruhrlandklinik/raw/train-test-2023",
        "event_mapping": {
            "wach": "wake",
            "n1": "n1",
            "n2": "n2",
            "n3": "n3",
            "rem": "rem",
        },
        "channels": [ChannelConfig("C4")],
        "grouped_channels": [
            ChannelConfig("F3", group="eeg"),
            ChannelConfig("F4", group="eeg"),
            ChannelConfig("C3", group="eeg"),
            ChannelConfig("C4", group="eeg"),
            ChannelConfig("O1", group="eeg"),
            ChannelConfig("O2", group="eeg"),
            ChannelConfig("M1", group="eeg"),
            ChannelConfig("M2", group="eeg"),
        ],
        "grouped_rereference": [["F3", "F4", "C3", "C4", "O1", "O2", "M1", "M2"]],
    },
    "ruhrland2024": {
        "clazz": Ruhrlandklinik,
        "edf_path": "ruhrlandklinik/raw/val-2024",
        "event_mapping": {
            "wach": "wake",
            "n1": "n1",
            "n2": "n2",
            "n3": "n3",
            "rem": "rem",
        },
        "channels": [ChannelConfig("C4")],
        "grouped_channels": [
            ChannelConfig("F3", group="eeg"),
            ChannelConfig("F4", group="eeg"),
            ChannelConfig("C3", group="eeg"),
            ChannelConfig("C4", group="eeg"),
            ChannelConfig("O1", group="eeg"),
            ChannelConfig("O2", group="eeg"),
            ChannelConfig("M1", group="eeg"),
            ChannelConfig("M2", group="eeg"),
        ],
        "grouped_rereference": [["F3", "F4", "C3", "C4", "O1", "O2", "M1", "M2"]],
    },
    "sleepedfx": {
        "clazz": SleepEDFx,
        "edf_path": "sleep-edfx",
        "event_mapping": {
            "sleep stage w": "wake",
            "sleep stage 1": "n1",
            "sleep stage 2": "n2",
            "sleep stage 3": "n3",
            "sleep stage 4": "n3",
            "sleep stage r": "rem",
        },
        "channels": [ChannelConfig("EEG Fpz-Cz")],
        "grouped_channels": [ChannelConfig("EEG Fpz-Cz", group="eeg"), ChannelConfig("EEG Pz-Oz", group="eeg")],
        "grouped_rereference": [["EEG Fpz-Cz", "EEG Pz-Oz"]],
    },
}

MODEL_CFG = {
    "sleeptransformer": {
        "clazz": SleepTransformer,
        "total_input": "630s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        "loss_function": torch.nn.functional.cross_entropy,
    },
    "sleeptransformer-aug": {
        "clazz": SleepTransformer,
        "total_input": "630s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        "loss_function": torch.nn.functional.cross_entropy,
        "transform": [
            RandomPolarityFlip(p=0.3, flip_p=0.1),
            RandomResampleJitter(p=0.5, scale=0.05),
        ],
    },
    "attnsleep": {
        "clazz": AttnSleep,
        "total_input": "30s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3, amsgrad=True),
        "loss_function": torch.nn.functional.cross_entropy,
        "loss_mode": "inverse-log",
        "class_weights": {"wake": 1.5, "n1": 2, "n2": 1.5, "n3": 1, "rem": 1.5},
    },
    "mrasleepnet": {
        "clazz": MRASleepNet,
        "total_input": "40s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=1e-4),
        "loss_function": torch.nn.functional.cross_entropy,
        "loss_mode": "inverse-log",
        "class_weights": {"wake": 1.5, "n1": 2, "n2": 1.5, "n3": 1, "rem": 1.5},
    },
    "seqsleepnet": {
        "clazz": SeqSleepNet,
        "total_input": "900s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=1e-4),
        "loss_function": torch.nn.functional.cross_entropy,
    },
    "tinysleepnet": {
        "clazz": TinySleepNet,
        "total_input": "500s",
        "total_input_model": "450s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1.0e-3),
        "loss_function": torch.nn.functional.cross_entropy,
        "loss_mode": "regular",
        "class_weights": {"wake": 1, "n1": 1.5, "n2": 1, "n3": 1, "rem": 1},
        "transform": [
            TimeShiftAndCrop(max_shift="60s", fill="closest", sampling_frequency=100, output_size="450s")
        ],
        "seq_len": 15,
        "use_lstm": True,
    },
    "usleep": {
        "clazz": USleep,
        "total_input": "1050s",
        "lr_scheduler": lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=5e-6, amsgrad=True),
        "loss_function": dice_loss,
        "loss_mode": "regular",
        "channel": [32, 64, 128, 256],
        "maxpool": [10, 8, 6, 4],
        "kernel": [5, 5, 5, 5],
        "activation": "relu",
        "norm": "batch",
        "mlp_size": 128,
        "balance_batches": True,
    },
}


def get_dataset(
    name: str,
    model_name: str,
    patients,
    path_root: str = DATASET_ROOT,
    grouped: bool = False,
    prepare_patient=None,
    prepare_target=None,
    prepare_sample=None,
    total_input: str | None = None,
):
    if name not in DATASET_CFG:
        raise ValueError(f"Unknown sleep staging dataset '{name}'.")
    if model_name not in MODEL_CFG:
        raise ValueError(f"Unknown sleep staging model '{model_name}'.")

    dataset_cfg = copy.deepcopy(DATASET_CFG[name])
    model_cfg = MODEL_CFG[model_name]
    sample_frequency = TASK_DEFAULTS["sample_frequency"]
    target_resolution = TASK_DEFAULTS["target_resolution"]
    selected_channels = dataset_cfg["grouped_channels"] if grouped else [dataset_cfg["channels"][0]]

    clazz = dataset_cfg["clazz"]
    edf_path = os.path.join(path_root, dataset_cfg["edf_path"])
    event_mapping = dataset_cfg["event_mapping"]
    if patients is None:
        patients = get_edf_files_in_repo(edf_path, recursive=True)
    rereference = dataset_cfg.get("grouped_rereference") if grouped else None

    return clazz(
        channels=[
            ChannelConfig(
                name=cfg.name,
                group=cfg.group,
                normalizer=EEGFilterNormalizer(fs=sample_frequency),
            )
            for cfg in selected_channels
        ],
        sample_frequency=sample_frequency,
        event_mapping=event_mapping,
        prepare_patient=prepare_patient,
        prepare_target=prepare_target,
        prepare_sample=prepare_sample,
        total_input=model_cfg["total_input"] if total_input is None else total_input,
        target_resolution=target_resolution,
        rereference=rereference,
    )


def get_model_and_trainer(
    name: str,
    dataset,
    epochs: int,
    n_samples: int,
    dry_run: bool = False,
    model_payload: dict[str, object] | None = None,
):
    if name not in MODEL_CFG:
        raise ValueError(f"Unknown sleep staging model '{name}'.")

    grad_reversal = bool((model_payload or {}).get("grad_reversal", False))
    if grad_reversal and dataset.get_n_datasets() < 3:
        raise ValueError("Gradient reversal requires more than two training datasets.")

    model_cfg = copy.deepcopy(MODEL_CFG[name])
    epochs = 1 if dry_run else epochs
    n_samples = 1_000 if dry_run else n_samples

    if "total_input_model" in model_cfg:
        freq = pd.to_timedelta(1.0 / dataset.sample_frequency, unit="s")
        total_input = pd.to_timedelta(model_cfg["total_input_model"])
        ts_len = int(total_input.total_seconds() / freq.total_seconds())
    else:
        ts_len = dataset.get_timeseries_len()

    clazz = model_cfg.pop("clazz")
    optimizer = model_cfg.pop("optimizer")
    lr_scheduler = model_cfg.pop("lr_scheduler")
    loss_function = model_cfg.pop("loss_function")
    loss_mode = model_cfg.pop("loss_mode", "regular")
    class_weights = model_cfg.pop("class_weights", {})
    balance_batches = model_cfg.pop("balance_batches", False)
    sig = inspect.signature(clazz)
    model_kwargs = {
        "classes": dataset.get_classes(),
        "ts_len": ts_len,
        "n_channels": len(dataset.channels),
        "sampling_frequency": dataset.sample_frequency,
        **model_cfg,
    }
    model = clazz(**{k: v for k, v in model_kwargs.items() if k in sig.parameters})

    trainer_kwargs = {
        "epochs": epochs,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "classes": dataset.get_classes(),
        "save_every": 1,
        "loss_function": loss_function,
        "early_stopping": 10,
        "train_transform": model_cfg.get("transform", None),
        "loss_mode": loss_mode,
        "class_weights": class_weights,
        "balance_batches": balance_batches,
    }
    if grad_reversal:
        trainer = GradReverseTrainer(
            feature_dim=512,
            n_domains=dataset.get_n_datasets(),
            **trainer_kwargs,
        )
    else:
        trainer = MulticlassTrainer(**trainer_kwargs)

    return model, trainer


def prepare_sleep_staging_patient(data_df, label_df, label_extra_df, patient=None):
    trimmed = trim_wake(data_df, label_df, label_extra_df)
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df

def list_filtered_sleep_patients(dataset_name: str, model_name: str, grouped: bool, dry_run: bool) -> list[str]:
    dataset_path = os.path.join(DATASET_ROOT, DATASET_CFG[dataset_name]["edf_path"])
    patients = get_edf_files_in_repo(dataset_path, recursive=True)
    dataset = get_dataset(
        name=dataset_name,
        model_name=model_name,
        patients=[],
        path_root=DATASET_ROOT,
        grouped=grouped,
        prepare_patient=prepare_sleep_staging_patient,
    )
    with suppress_stdout_logging(logger):
        filtered = filter_patients_by_sleep_time(
            patients=patients,
            dataset=dataset,
            sleep_labels=SCORED_SLEEP_LABELS,
            quantile=SLEEP_TIME_FILTER_QUANTILE,
            num_workers=NUM_WORKERS_DATASET,
            label=dataset_name,
        )
    
    logger.info(
        f"Filtered sleep-time outliers: kept {len(filtered)}/{len(patients)} patients"
    )
    return filtered

def load_split_with_logging(
    dataset_name: str,
    purpose: str,
    model_name: str,
    grouped: bool,
    patients: list[str],
):
    with suppress_stdout_logging(logger):
        logger.context(purpose.upper())
        try:
            total_input = None
            if purpose == "test":
                total_input = MODEL_CFG[model_name].get("total_input_model", MODEL_CFG[model_name]["total_input"])

            dataset = get_dataset(
                name=dataset_name,
                model_name=model_name,
                patients=patients,
                path_root=DATASET_ROOT,
                grouped=grouped,
                prepare_patient=prepare_sleep_staging_patient,
                prepare_target=partial(prepare_multiclass_target, target_classes=TARGET_CLASSES),
                prepare_sample=None,
                total_input=total_input,
            )
            dataset.initialize(patients, NUM_WORKERS_DATASET)
            return dataset
        finally:
            logger.uncontext()


def build_splits_for_dataset(dataset_name: str, args):
    patients = list_filtered_sleep_patients(dataset_name, args.model, args.grouped, args.dry)
    if args.test is not None:
        train_patients, val_patients = random_split(patients, test_frac=VAL_FRAC)
        train_ds = load_split_with_logging(dataset_name, "train", args.model, args.grouped, train_patients)
        val_ds = load_split_with_logging(dataset_name, "val", args.model, args.grouped, val_patients)
        return train_ds, val_ds, None

    train_patients, val_patients, test_patients = split_patients_train_val_test(patients, TEST_FRAC, VAL_FRAC)
    train_ds = load_split_with_logging(dataset_name, "train", args.model, args.grouped, train_patients)
    val_ds = load_split_with_logging(dataset_name, "val", args.model, args.grouped, val_patients)
    test_ds = load_split_with_logging(dataset_name, "test", args.model, args.grouped, test_patients)
    return train_ds, val_ds, test_ds


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a sleep-staging model.")
    parser.add_argument("--model", required=False, default="sleeptransformer", type=str)
    parser.add_argument("--train", required=False, nargs="+", default=["sleepedfx"], type=str)
    parser.add_argument("--test", required=False, nargs="*", default=None, type=str)
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--gradrev", action="store_true")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    train_parts = []
    val_parts = []
    test_parts = []

    if args.test is not None:
        for dataset_name in args.train:
            train_ds, val_ds, _ = build_splits_for_dataset(dataset_name, args)
            train_parts.append(train_ds)
            val_parts.append(val_ds)

        for dataset_name in args.test:
            patients = list_filtered_sleep_patients(dataset_name, args.model, args.grouped, args.dry)
            test_ds = load_split_with_logging(dataset_name, "test", args.model, args.grouped, patients)
            test_parts.append((dataset_name, test_ds))
    else:
        for dataset_name in args.train:
            train_ds, val_ds, test_ds = build_splits_for_dataset(dataset_name, args)
            train_parts.append(train_ds)
            val_parts.append(val_ds)
            test_parts.append((dataset_name, test_ds))

    train_dataset = combine_datasets(train_parts)
    model, trainer = get_model_and_trainer(
        args.model,
        train_dataset,
        EPOCHS,
        N_SAMPLES,
        args.dry,
        {"grad_reversal": args.gradrev},
    )
    experiment_name = f"{EXPERIMENT_NAME}-grouped" if args.grouped else EXPERIMENT_NAME
    if args.gradrev:
        experiment_name += "-gradrev"
    if args.dry:
        logger.info("Performing dry run to test pipeline!")
        experiment_name += "-dev"

    collate_ignore = ["time", "patient", "dataset"] if hasattr(train_dataset, "datasets") else ["time", "patient"]
    run(
        RunCfg(
            experiment_name=experiment_name,
            model_name=f"{args.model}{'_GradReversal' if args.gradrev else ''}",
            model=model,
            trainer=trainer,
            train_datasets=train_parts,
            val_datasets=val_parts,
            test_datasets=test_parts,
            batch_size=BATCH_SIZE,
            n_samples=1_000 if args.dry else N_SAMPLES,
            num_workers_dataloader=NUM_WORKERS_DATALOADER,
            test_repeats=GROUPED_TEST_REPEATS if args.grouped else [1],
            use_energy_tracker=False, #not args.dry,
            tags={"model": args.model, **({"trainer": "gradrev"} if args.gradrev else {})},
            collate_fn=partial(batch_collate, ignore_list=collate_ignore),
        ),
    )


if __name__ == "__main__":
    main()
