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

from sleepwalker.datasets import ChannelConfig
from sleepwalker.datasets.ABC import ABC
from sleepwalker.datasets.Apples import Apples
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.CAP import CAP
from sleepwalker.datasets.ISRUC import ISRUC
from sleepwalker.datasets.MNC import MNC
from sleepwalker.datasets.MROS import MROS
from sleepwalker.datasets.NCHSDB import NCHSDB
from sleepwalker.datasets.Ruhrlandklinik import RuhrlandklinikDiagnosis, get_channels
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
from sleepwalker.trainer.DiagnosisTrainer import DiagnosisTrainer
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.losses import dice_loss
from sleepwalker.trainer.utils.filtering import trim_event
from sleepwalker.datasets.MultiDataset import combine_datasets
from sleepwalker.trainer.utils.targets import prepare_multiclass_target, prepare_diagnosis_target
from sleepwalker.utils import logger, suppress_stdout_logging

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
mp.set_sharing_strategy("file_system")

# TARGET_CLASSES = ["wake", "n1", "n2", "n3", "rem"] # Target classes are dependent on the dataset
SLEEP_LABELS = ["n1", "n2", "n3", "rem"]

SAMPLE_FREQUENCY = 100
TARGET_RESOLUTION = "300s"
SEQUENCES_PER_PATIENT = 10
BATCH_SIZE = 128
EPOCHS = 100
N_SAMPLES = 25_000
NUM_WORKERS_DATASET = 8
NUM_WORKERS_DATALOADER = 8
TEST_FRAC = 0.3
VAL_FRAC = 0.1
EXPERIMENT_NAME = "diagnosis classification"
DATASET_ROOT = "/raid/wald/sleepwalker"
GROUPED_TEST_REPEATS = [1, 2, 3, 4, 5, 10]
SLEEP_TIME_FILTER_QUANTILE = 0.05

DATASET_CFG = {
    "shhs": {
        "clazz": SHHS,
        "edf_path": "shhs",
        "diagnosis_label_mapping": {
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
    "ruhrland": {
        "clazz": RuhrlandklinikDiagnosis,
        "edf_path": "ruhrlandklinik/raw",
        "diagnosis_label_mapping": {
            "healthy": "healthy",
            "osa": "osa",
            "osa_plm": "osa_plm",
            "plm": "plm",
            "plm_uars": "plm_uars",
            "uars": "uars"
        },
        "channels": ["C4-M1", "RIP Flow", "Saturation", "Right Leg"],
        "grouped_channels": None,
    },
    "ruhrland2023": {
        "clazz": RuhrlandklinikDiagnosis,
        "edf_path": "ruhrlandklinik/raw/train-test-2023",
        "diagnosis_label_mapping": {
            "healthy": "healthy",
            "osa": "osa",
            "osa_plm": "osa_plm",
            "plm": "plm",
            "plm_uars": "plm_uars",
            "uars": "uars"
        },
        "channels": ["C4-M1", "RIP Flow", "Saturation", "Right Leg"],
        "grouped_channels": None,
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
        "total_input": "300s",
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



def build_dataset(dataset_name: str, patients: list[str], grouped: bool, total_input: str, m_sequences: int):
    if dataset_name not in DATASET_CFG:
        raise ValueError(f"Unknown sleep staging dataset '{dataset_name}'.")

    dataset_cfg = copy.deepcopy(DATASET_CFG[dataset_name])
    target_classes = list(DATASET_CFG[dataset_name]["diagnosis_label_mapping"].values())
    channels = DATASET_CFG[dataset_name]["grouped_channels"] if grouped else DATASET_CFG[dataset_name]["channels"]
    channels = get_channels(channels, grouped=grouped, include_quality=False, normalize=True, sample_frequency=SAMPLE_FREQUENCY) # NOTE: need to override any channels?
    dataset = dataset_cfg["clazz"](
        channels=channels,
        sample_frequency=SAMPLE_FREQUENCY,
        sequences_per_patient=m_sequences,
        diagnosis_label_mapping=dataset_cfg["diagnosis_label_mapping"],
        prepare_patient=None,
        prepare_target=partial(prepare_diagnosis_target, target_classes=target_classes),
        prepare_sample=None,
        total_input=total_input,
        # rereference=rereference,
    )
    
    dataset.initialize(patients, NUM_WORKERS_DATASET)
    dataset.classes = target_classes
    logger.info(f"{dataset_name}: loaded {len(patients)} patients")
    return dataset


def list_patients(dataset_name: str, grouped: bool, dry_run: bool) -> list[str]:
    dataset_path = os.path.join(DATASET_ROOT, DATASET_CFG[dataset_name]["edf_path"])
    patients = get_edf_files_in_repo(dataset_path, recursive=True)
    if dry_run:
        return patients[:5]
    channels = DATASET_CFG[dataset_name]["grouped_channels"] if grouped else DATASET_CFG[dataset_name]["channels"]
    channels = get_channels(channels, grouped=grouped, include_quality=False, normalize=True, sample_frequency=SAMPLE_FREQUENCY)
    # TODO: implement get_channels for all diagnosis datasets
    dataset = DATASET_CFG[dataset_name]["clazz"](
        channels=channels,
        sample_frequency=SAMPLE_FREQUENCY,
        diagnosis_label_mapping=DATASET_CFG[dataset_name]["diagnosis_label_mapping"],
        prepare_patient=None,
        prepare_target=None,
        prepare_sample=None,
        total_input="30s",
        # rereference=rereference,
    )
    # with suppress_stdout_logging(logger):
    #     filtered = filter_patients_by_sleep_time(
    #         patients=patients,
    #         dataset=dataset,
    #         quantile=SLEEP_TIME_FILTER_QUANTILE,
    #         num_workers=NUM_WORKERS_DATASET,
    #         label=dataset_name,
    #     )
    return patients # filtered


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

    trainer = DiagnosisTrainer(
        epochs=epochs,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        classes=train_dataset.get_classes(),
        sequences_per_patient=train_dataset.sequences_per_patient,
        save_every=0,
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

    # with suppress_stdout_logging(logger):
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
        logger.warning(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}, Test patients: {len(test_patients)}")

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
        train_parts.append(build_dataset(dataset_name, train_patients, args.grouped, train_total_input, args.m_sequences))
        logger.uncontext()

    for dataset_name, val_patients in val_patient_splits:
        logger.context(f"VAL:{dataset_name}")
        val_parts.append(build_dataset(dataset_name, val_patients, args.grouped, test_total_input, args.m_sequences))
        logger.uncontext()

    for dataset_name, test_patients in test_patient_splits:
        if len(test_patients) == 0:
            raise ValueError(f"Test split for {dataset_name} is empty.")
        logger.context(f"TEST:{dataset_name}")
        test_parts.append((dataset_name, build_dataset(dataset_name, test_patients, args.grouped, test_total_input, args.m_sequences)))
        logger.uncontext()

    logger.warning(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}, Test patients: {len(test_patients)}")
    logger.warning(f"Train dataset size: {train_parts[0].get_n_patients()}")


    return train_parts, val_parts, test_parts


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a sleep-staging model.")
    parser.add_argument("--model", type=str, default="usleep", choices=sorted(MODEL_CFG))
    parser.add_argument("--train", nargs="+", default=["ruhrland2023"], choices=sorted(DATASET_CFG))
    parser.add_argument("--test", nargs="*", default=[], choices=sorted(DATASET_CFG))
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--m_sequences", type=int, default=SEQUENCES_PER_PATIENT, help="Number of sequences to sample per patient.")
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

    os.makedirs(os.path.join("results", "diagnosis", experiment_name), exist_ok=True)
    logger.set_log_file(os.path.join("results", "diagnosis", experiment_name, "output.log"))

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
            log_path=os.path.join("results", "diagnosis"),
            tags={"model": args.model},
            collate_fn=partial(batch_collate, ignore_list=collate_ignore),
            meta_data=vars(args),
        )
    )


if __name__ == "__main__":
    main()
