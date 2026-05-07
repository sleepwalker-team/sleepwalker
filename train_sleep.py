#!/bin/env python3

from __future__ import annotations

import argparse
import copy
import inspect
import os
from functools import partial

import pandas as pd

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import torch
import torch.multiprocessing as mp

from sleepwalker.datasets import ChannelConfig, Ruhrlandklinik
from sleepwalker.datasets.ABC import ABC
from sleepwalker.datasets.Apples import Apples
from sleepwalker.datasets.Basedataset import batch_collate
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
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.losses import dice_loss
from sleepwalker.trainer.utils.filtering import trim_event
from sleepwalker.datasets.MultiDataset import combine_datasets
from sleepwalker.trainer.utils.targets import prepare_multiclass_target
from sleepwalker.utils import logger, suppress_stdout_logging

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
mp.set_sharing_strategy("file_system")

TARGET_CLASSES = ["wake", "n1", "n2", "n3", "rem"]
SLEEP_LABELS = ["n1", "n2", "n3", "rem"]

SAMPLE_FREQUENCY = 100
TARGET_RESOLUTION = "30s"
BATCH_SIZE = 128
EPOCHS = 100
N_SAMPLES = 250_000
NUM_WORKERS_DATASET = 8
NUM_WORKERS_DATALOADER = 8
TEST_FRAC = 0.3
VAL_FRAC = 0.1
EXPERIMENT_NAME = "sleep"
DATASET_ROOT = "/raid/sleepwalker"
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


def prepare_sleep_staging_patient(data_df, label_df, label_extra_df, patient=None):
    trimmed = trim_event(data_df, label_df, label_extra_df, SLEEP_LABELS)
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df


def summarize_patient_sleep_time(patient: str, label_df: pd.DataFrame | None, **_kwargs) -> dict[str, float | str] | None:
    if label_df is None or len(label_df) == 0:
        return None

    durations = label_df.copy()
    durations["duration_s"] = (durations["Endtime"] - durations["Starttime"]).dt.total_seconds()
    sleep_seconds = durations.loc[durations["Label"].isin(SLEEP_LABELS), "duration_s"].sum()
    return {"patient": patient, "sleep_seconds": float(sleep_seconds)}


def filter_patients_by_sleep_time(
    patients: list[str],
    dataset,
    quantile: float,
    num_workers: int,
    label: str,
) -> list[str]:
    if len(patients) < 3:
        return patients

    stats_df = dataset.get_patient_stats(
        patients,
        summarize_patient_sleep_time,
        num_workers=num_workers,
    )
    if len(stats_df) < 3 or "sleep_seconds" not in stats_df.columns:
        return patients

    lower = stats_df["sleep_seconds"].quantile(quantile)
    upper = stats_df["sleep_seconds"].quantile(1 - quantile)
    filtered = stats_df[(stats_df["sleep_seconds"] >= lower) & (stats_df["sleep_seconds"] <= upper)]["patient"].tolist()
    logger.info(
        f"Filtered sleep-time outliers for {label}: kept {len(filtered)}/{len(patients)} patients "
        f"after dropping the bottom/top {quantile:.0%}."
    )
    return filtered


def build_channel_configs(dataset_name: str, grouped: bool) -> tuple[list[ChannelConfig], list[list[str]] | None]:
    dataset_cfg = DATASET_CFG[dataset_name]
    selected_channels = dataset_cfg["grouped_channels"] if grouped else [dataset_cfg["channels"][0]]
    channels = [
        ChannelConfig(
            name=cfg.name,
            group=cfg.group,
            normalizer=EEGFilterNormalizer(fs=SAMPLE_FREQUENCY),
        )
        for cfg in selected_channels
    ]
    rereference = dataset_cfg.get("grouped_rereference") if grouped else None
    return channels, rereference


def build_dataset(dataset_name: str, patients: list[str], grouped: bool, total_input: str):
    if dataset_name not in DATASET_CFG:
        raise ValueError(f"Unknown sleep staging dataset '{dataset_name}'.")

    dataset_cfg = copy.deepcopy(DATASET_CFG[dataset_name])
    channels, rereference = build_channel_configs(dataset_name, grouped)
    dataset = dataset_cfg["clazz"](
        channels=channels,
        sample_frequency=SAMPLE_FREQUENCY,
        event_mapping=dataset_cfg["event_mapping"],
        prepare_patient=prepare_sleep_staging_patient,
        prepare_target=partial(prepare_multiclass_target, target_classes=TARGET_CLASSES),
        prepare_sample=None,
        total_input=total_input,
        target_resolution=TARGET_RESOLUTION,
        rereference=rereference,
    )
    dataset.classes = list(TARGET_CLASSES)
    dataset.initialize(patients, NUM_WORKERS_DATASET)
    logger.info(f"{dataset_name}: loaded {len(patients)} patients")
    return dataset


def list_patients(dataset_name: str, grouped: bool, dry_run: bool) -> list[str]:
    dataset_path = os.path.join(DATASET_ROOT, DATASET_CFG[dataset_name]["edf_path"])
    patients = get_edf_files_in_repo(dataset_path, recursive=True)
    if dry_run:
        return patients[:2]

    channels, rereference = build_channel_configs(dataset_name, grouped)
    dataset = DATASET_CFG[dataset_name]["clazz"](
        channels=channels,
        sample_frequency=SAMPLE_FREQUENCY,
        event_mapping=DATASET_CFG[dataset_name]["event_mapping"],
        prepare_patient=prepare_sleep_staging_patient,
        prepare_target=None,
        prepare_sample=None,
        total_input="30s",
        target_resolution=TARGET_RESOLUTION,
        rereference=rereference,
    )
    with suppress_stdout_logging(logger):
        filtered = filter_patients_by_sleep_time(
            patients=patients,
            dataset=dataset,
            quantile=SLEEP_TIME_FILTER_QUANTILE,
            num_workers=NUM_WORKERS_DATASET,
            label=dataset_name,
        )
    return filtered


def build_model_and_trainer(train_dataset, model_name: str, dry_run: bool):
    if model_name not in MODEL_CFG:
        raise ValueError(f"Unknown sleep staging model '{model_name}'.")

    model_cfg = copy.deepcopy(MODEL_CFG[model_name])
    epochs = 2 if dry_run else EPOCHS

    total_input_model = model_cfg.pop("total_input_model", None)
    model_cfg.pop("total_input")
    train_transform = model_cfg.pop("transform", None)
    clazz = model_cfg.pop("clazz")
    optimizer = model_cfg.pop("optimizer")
    lr_scheduler = model_cfg.pop("lr_scheduler")
    loss_function = model_cfg.pop("loss_function")
    loss_mode = model_cfg.pop("loss_mode", "regular")
    class_weights = model_cfg.pop("class_weights", {})
    balance_batches = model_cfg.pop("balance_batches", False)

    # TODO: use get_input_spec?
    if total_input_model is not None:
        freq = pd.to_timedelta(1.0 / train_dataset.sample_frequency, unit="s")
        total_input = pd.to_timedelta(total_input_model)
        ts_len = int(total_input.total_seconds() / freq.total_seconds())
    else:
        ts_len = train_dataset.get_timeseries_len()

    sig = inspect.signature(clazz)
    model_kwargs = {
        "classes": train_dataset.get_classes(),
        "ts_len": ts_len,
        "n_channels": len(train_dataset.get_input_channels()),
        "sampling_frequency": train_dataset.sample_frequency,
        **model_cfg,
    }
    model = clazz(**{k: v for k, v in model_kwargs.items() if k in sig.parameters})

    trainer = MulticlassTrainer(
        epochs=epochs,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        classes=train_dataset.get_classes(),
        save_every=1,
        loss_function=loss_function,
        early_stopping=15,
        train_transform=train_transform,
        loss_mode=loss_mode,
        class_weights=class_weights,
        balance_batches=balance_batches,
    )
    return model, trainer


def build_dataset_parts(args, train_total_input: str, test_total_input: str):
    train_parts = []
    val_parts = []
    test_parts = []

    train_patient_splits: list[tuple[str, list[str]]] = []
    val_patient_splits: list[tuple[str, list[str]]] = []
    test_patient_splits: list[tuple[str, list[str]]] = []

    with suppress_stdout_logging(logger):
        total_train_patients = 0
        total_val_patients = 0

        for dataset_name in args.train:
            patients = list_patients(dataset_name, args.grouped, args.dry)
            if len(patients) == 0:
                raise ValueError(f"No patients found for {dataset_name}.")

            if args.val_frac > 0:
                remaining_patients, val_patients = random_split(patients, test_frac=args.val_frac)
            else:
                remaining_patients, val_patients = list(patients), []

            if len(args.test) == 0 and args.test_frac > 0:
                train_patients, test_patients = random_split(remaining_patients, test_frac=args.test_frac)
            else:
                train_patients, test_patients = list(remaining_patients), []

            total_train_patients += len(train_patients)
            total_val_patients += len(val_patients)
            train_patient_splits.append((dataset_name, train_patients))
            if len(val_patients) > 0:
                val_patient_splits.append((dataset_name, val_patients))
            if len(args.test) == 0:
                test_patient_splits.append((dataset_name, test_patients))

        if len(args.test) > 0:
            for dataset_name in args.test:
                patients = list_patients(dataset_name, args.grouped, args.dry)
                if len(patients) == 0:
                    raise ValueError(f"No patients found for {dataset_name}.")
                test_patient_splits.append((dataset_name, patients))

        for dataset_name, train_patients in train_patient_splits:
            logger.context(f"TRAIN:{dataset_name}")
            train_parts.append(build_dataset(dataset_name, train_patients, args.grouped, train_total_input))
            logger.uncontext()

        for dataset_name, val_patients in val_patient_splits:
            logger.context(f"VAL:{dataset_name}")
            val_parts.append(build_dataset(dataset_name, val_patients, args.grouped, test_total_input))
            logger.uncontext()

        for dataset_name, test_patients in test_patient_splits:
            if len(test_patients) == 0:
                raise ValueError(f"Test split for {dataset_name} is empty.")
            logger.context(f"TEST:{dataset_name}")
            test_parts.append((dataset_name, build_dataset(dataset_name, test_patients, args.grouped, test_total_input)))
            logger.uncontext()

    return train_parts, val_parts, test_parts


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a sleep-staging model.")
    parser.add_argument("--model", type=str, default="sleeptransformer", choices=sorted(MODEL_CFG))
    parser.add_argument("--train", nargs="+", default=["sleepedfx"], choices=sorted(DATASET_CFG))
    parser.add_argument("--test", nargs="*", default=[], choices=sorted(DATASET_CFG))
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--id", type=str, default="", help="ID of the experiment.")
    parser.add_argument("--val_frac", type=float, default=VAL_FRAC, help="Fraction of train patients reserved for validation.")
    parser.add_argument("--test_frac", type=float, default=TEST_FRAC, help="Fraction reserved for internal test splits.")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    experiment_name = f"{EXPERIMENT_NAME}_{args.id}" if args.id else EXPERIMENT_NAME
    if args.grouped:
        experiment_name += "-grouped"
    if args.dry:
        logger.info("Performing dry run to test pipeline!")
        experiment_name += "-dev"

    os.makedirs(os.path.join("results", "sleep", experiment_name), exist_ok=True)
    logger.set_log_file(os.path.join("results", "sleep", experiment_name, "output.log"))

    model_cfg = MODEL_CFG[args.model]
    train_total_input = model_cfg["total_input"]
    test_total_input = model_cfg.get("total_input_model", train_total_input)

    train_parts, val_parts, test_parts = build_dataset_parts(args, train_total_input, test_total_input)
    train_dataset = combine_datasets(train_parts)
    model, trainer = build_model_and_trainer(train_dataset, args.model, args.dry)

    collate_ignore = ["time", "patient", "dataset"] if hasattr(train_dataset, "datasets") else ["time", "patient"]
    run(
        RunCfg(
            experiment_name=experiment_name,
            model_name=args.model,
            model=model,
            trainer=trainer,
            train_datasets=train_parts,
            val_datasets=val_parts,
            test_datasets=test_parts,
            batch_size=BATCH_SIZE,
            n_samples=1_000 if args.dry else N_SAMPLES,
            num_workers_dataloader=NUM_WORKERS_DATALOADER,
            test_repeats=GROUPED_TEST_REPEATS if args.grouped else [1],
            use_energy_tracker=False,
            use_mlflow=True,
            log_path=os.path.join("results", "sleep"),
            tags={"model": args.model},
            collate_fn=partial(batch_collate, ignore_list=collate_ignore),
            meta_data=vars(args),
        )
    )


if __name__ == "__main__":
    main()
