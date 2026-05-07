#!/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
import os
from functools import partial

import pandas as pd
from sleepwalker.models.preprocessors.RobustScaler import RobustScaler

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import torch
import torch.multiprocessing as mp

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets import ChannelConfig, Ruhrlandklinik
from sleepwalker.datasets.Ruhrlandklinik import get_channels
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.normalizer.PulseFilterNormalizer import PulseFilterNormalizer
from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer
from sleepwalker.datasets.utils import get_edf_files_in_repo, random_split
from sleepwalker.models import MultiModel, MetaModelEntry, SleepTransformer
from sleepwalker.models.UTime import UTime
from sleepwalker.deployment import export_prediction_package
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.utils.filtering import trim_event
from sleepwalker.trainer.utils.targets import prepare_multiclass_target
from sleepwalker.utils import logger, suppress_stdout_logging

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
mp.set_sharing_strategy("file_system")

TRAIN_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/train-test-2023"
TEST_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/val-2024"
BATCH_SIZE = 64
EPOCHS = 50
N_SAMPLES = 100_000
NUM_WORKERS_DATASET = 8
NUM_WORKERS_DATALOADER = 8
SAMPLE_FREQUENCY = 25
TARGET_RESOLUTION = "5s" 
STRIDE = "1s"            
TARGET_CLASSES = ["apnea", "hypopnea", "regular breathing"]
MIN_DESAT_SLEEP_OVERLAP_S = 1.0

def _clip_to_intervals(df: pd.DataFrame | None, intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame | None:
    if df is None or len(df) == 0 or len(intervals) == 0:
        return None

    clipped_rows = []
    for row in df.itertuples(index=False):
        row_start = pd.Timestamp(row.Starttime)
        row_end = pd.Timestamp(row.Endtime)
        for keep_start, keep_end in intervals:
            clipped_start = max(row_start, keep_start)
            clipped_end = min(row_end, keep_end)
            if clipped_start >= clipped_end:
                continue
            clipped_row = dict(zip(df.columns, row))
            clipped_row["Starttime"] = clipped_start
            clipped_row["Endtime"] = clipped_end
            clipped_rows.append(clipped_row)

    if len(clipped_rows) == 0:
        return None

    return pd.DataFrame(clipped_rows, columns=df.columns)

def prepare_patient(data_df, label_df, label_extra_df, patient=None):
    trimmed = trim_event(data_df, label_df, label_extra_df,["sleep"])
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed

    trimmed = trim_event(data_df, label_df, label_extra_df, ["desaturation"])
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed

    df = label_df.copy()
    df["Starttime"] = pd.to_datetime(df["Starttime"])
    df["Endtime"] = pd.to_datetime(df["Endtime"])

    desat = df[df["Label"].eq("desaturation")].sort_values(["Starttime", "Endtime"])
    sleep = df[df["Label"].eq("sleep")].sort_values(["Starttime", "Endtime"])
    if len(desat) == 0 or len(sleep) == 0:
        return None

    overlap_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    sleep_rows = list(sleep.itertuples(index=False))
    sleep_idx = 0
    for desat_row in desat.itertuples(index=False):
        desat_start = pd.Timestamp(desat_row.Starttime)
        desat_end = pd.Timestamp(desat_row.Endtime)

        while sleep_idx < len(sleep_rows) and pd.Timestamp(sleep_rows[sleep_idx].Endtime) <= desat_start:
            sleep_idx += 1

        cur_idx = sleep_idx
        while cur_idx < len(sleep_rows):
            sleep_start = pd.Timestamp(sleep_rows[cur_idx].Starttime)
            sleep_end = pd.Timestamp(sleep_rows[cur_idx].Endtime)
            if sleep_start >= desat_end:
                break

            overlap_start = max(desat_start, sleep_start)
            overlap_end = min(desat_end, sleep_end)
            if overlap_end > overlap_start and (overlap_end - overlap_start).total_seconds() >= MIN_DESAT_SLEEP_OVERLAP_S:
                overlap_intervals.append((overlap_start, overlap_end))
            cur_idx += 1

    if len(overlap_intervals) == 0:
        return None

    label_df = _clip_to_intervals(df, overlap_intervals)
    if label_df is None or len(label_df) == 0:
        return None

    return data_df, label_df, label_extra_df

def prepare_sample(
    data,
    quality_data=None,
    target=None,
    target_extra=None,
    patient=None,
    time=None,
):
    item = {
        "data": torch.from_numpy(data.values).float(),
        "target": target,
        "patient": patient,
        "time": time,
    }
    if target_extra is not None:
        item["target_extra"] = target_extra
    return item

def build_dataset(patients: list[str], channels: list[str], total_input:str, include_pap:bool):
    EVENT_MAPPING = {
        "entsättigung": "desaturation",
        "wach":"wake",
        "n1": "sleep",
        "n2": "sleep",
        "n3": "sleep",
        "rem": "sleep",
        "a. gemischt": "apnea",
        "a. obstruktiv": "apnea",
        "a. zentral": "apnea",
        "apnoe": "apnea",
        "h. obstruktiv": "hypopnea",
        "h. zentral": "hypopnea",
        "hypopnea-gemischt": "hypopnea",
        "hypopnoe": "hypopnea",
    }

    if include_pap:
        EVENT_MAPPING["fg apnoe geschlossen"] = "apnea"
        EVENT_MAPPING["fg apnoe geöffnet"] = "apnea"
        EVENT_MAPPING["fg-apnoe unbekannt"] = "apnea"
        EVENT_MAPPING["fg hypopnoe"] = "hypopnea"

    prepare_breathing_target = partial(
        prepare_multiclass_target,
        target_classes=TARGET_CLASSES,
        filters=[
            {"columns": ["desaturation"], "percentage": 0.5, "mode": "min"},
            {"columns": ["sleep"], "percentage": 0.5, "mode": "min"},
        ],
    )

    channel_configs = get_channels(
        channels,
        grouped=False,
        include_quality=False,
        normalize=True,
        sample_frequency=SAMPLE_FREQUENCY,
    )
    dataset = Ruhrlandklinik(
        channels=channel_configs,
        sample_frequency=SAMPLE_FREQUENCY,
        event_mapping=EVENT_MAPPING,
        stride=STRIDE,
        prepare_patient=prepare_patient,
        prepare_target=prepare_breathing_target,
        prepare_sample=prepare_sample,
        total_input=total_input,
        target_resolution=TARGET_RESOLUTION
    )
    dataset.classes = list(TARGET_CLASSES)
    logger.info(
        f"Configured {len(dataset.get_input_channels())} effective input channels "
        f"from {len(channel_configs)} Ruhrland channel candidates."
    )
    dataset.initialize(patients, NUM_WORKERS_DATASET)
    return dataset

def is_pap_patient(edf_path: str) -> bool:
    meta = read_edf_meta(edf_path)
    available_channels = meta["signals"]
    pap_channel_patterns = [
        "Druckeinstellung",
        "EPAP",
        "IPAP",
        "Druck (PAP)",
        "Mask Pressure",
        "Leck (PAP)",
        "Fluss (PAP)",
        "SpO2 (PAP)",
        "Puls (PAP)",
        "FiO2 (PAP)",
        "PrismaLeak",
        "PrismaFlow",
        "AutoPressure",
        "Ventilation Vorg",
        "AchievedAlveolar",
    ]
    return any(ch in available_channels for ch in pap_channel_patterns)


def has_required_channels(edf_path: str, channels:list[str]) -> bool:
    meta = read_edf_meta(edf_path)
    available_channels = set(meta["signals"])
    requested = get_channels(channels, grouped=False, include_quality=False, normalize=False, sample_frequency=SAMPLE_FREQUENCY) 
    channel_by_group = defaultdict(list)
    for cfg in requested:
        if cfg.group:
            channel_by_group[cfg.group].append(cfg.name)
        else:
            channel_by_group[cfg.name].append(cfg.name)

    for requested_channels in channel_by_group.values():
        if not any(channel in available_channels for channel in requested_channels):
            return False
    return True

def list_patients(source_root: str, channels: list[str], dry_run: bool, pap: bool) -> list[str]:
    patients = [
        patient for patient in get_edf_files_in_repo(source_root, recursive=True) if has_required_channels(patient, channels) 
    ]

    if not pap:
        patients = [p for p in patients if not is_pap_patient(p)]

    return patients[:2] if dry_run else patients

def build_model_and_trainer(train_dataset, model, scaler, class_weights, dry):
    n_channels = len(train_dataset.get_input_channels())
    if model == "utime-big":
        model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=n_channels,
            classes=TARGET_CLASSES, 
            sampling_frequency=SAMPLE_FREQUENCY,
            channel = [64, 128, 128, 256],
            kernel = [5, 5, 3, 3],
            maxpool = [5, 5, 3, 3],
            norm="channel",
            activation="elu",
            mlp_size=512,
            dropout_p=0,
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(0, n_channels)])] if scaler else None
        )
    elif model == "utime-huge":
        model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=n_channels,
            classes=TARGET_CLASSES, 
            sampling_frequency=SAMPLE_FREQUENCY,
            channel = [32, 64, 128, 128, 256],
            kernel = [5, 5, 3, 3, 3],
            maxpool = [5, 5, 3, 3, 3],
            norm="channel",
            activation="elu",
            mlp_size=1024,
            dropout_p=0,
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(0, n_channels)])] if scaler else None
        )
    elif model == "utime-small":
        model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=n_channels,
            classes=TARGET_CLASSES, 
            sampling_frequency=SAMPLE_FREQUENCY,
            channel = [64, 128, 256],
            kernel = [5, 3, 3],
            maxpool = [5, 3, 3],
            norm="channel",
            activation="elu",
            mlp_size=64,
            dropout_p=0,
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(0, n_channels)])] if scaler else None
        )
    else:
        raise ValueError(f"Did not recoginize model {model}")
    
    trainer = MulticlassTrainer(
        epochs=EPOCHS if not dry else 2,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ), 
        classes=TARGET_CLASSES,
        loss_function=torch.nn.functional.cross_entropy,#torch.nn.functional.binary_cross_entropy_with_logits,
        save_every=10,
        #loss_mode="inverse",
        balance_batches=True,
        balance_gamma=0.75,
        class_weights=class_weights
    )
    return model, trainer


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an breathing model.")
    parser.add_argument("--channels", nargs='+', default=["RIP Flow", "RIP Sum", "Chest", "Abdomen", "Saturation"], help="List of channels.")
    parser.add_argument("--scaler", action="store_true", help="If RobustScaler should be used.")
    parser.add_argument("--pap", action="store_true", help="Include PAP patients.")
    parser.add_argument("--apnea_weight", type=int, default=1, help="Weight of breathing classes (apnea/hypopnea).")
    parser.add_argument("--hypopnea_weight", type=int, default=1, help="Weight of breathing classes (apnea/hypopnea).")
    parser.add_argument("--model", type=str, default="utime-big", help="What model to use.")
    parser.add_argument("--id", type=str, default="", help="ID of the experiment")
    parser.add_argument("--total_input", type=str, default="60s", help="Total input size")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Fraction of train patients reserved for validation. Set to 0 to disable.")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    experiment_name = f"breathing_{args.id}" 
    if args.dry:
        logger.info("Performing dry run to test pipeline!")
        experiment_name += "-dev"
    
    os.makedirs(os.path.join("results", "breathing", experiment_name), exist_ok=True)
    logger.set_log_file(os.path.join("results", "breathing", experiment_name, "output.log"))

    train_patients = list_patients(TRAIN_ROOT, args.channels, args.dry, args.pap)
    test_patients = list_patients(TEST_ROOT, args.channels, args.dry, args.pap)

    val_dataset = None
    if args.val_frac is not None and args.val_frac > 0:
        train_patients, val_patients = random_split(train_patients, test_frac=args.val_frac)
    else:
        val_patients = []

    # with suppress_stdout_logging(logger):
    logger.context("TRAIN")
    train_dataset = build_dataset(
        train_patients,
        args.channels,
        args.total_input,
        args.pap
    )
    logger.uncontext()
    if len(val_patients) > 0:
        logger.context("VAL")
        val_dataset = build_dataset(
            val_patients,
            args.channels,
            args.total_input,
            args.pap
        )
        logger.uncontext()
    logger.context("TEST")
    test_dataset = build_dataset(
        test_patients,
        args.channels,
        args.total_input,
        args.pap
    )
    logger.uncontext()

    model, trainer = build_model_and_trainer(
        train_dataset,
        args.model,
        args.scaler,
        {"apnea":args.apnea_weight, "hypopnea":args.hypopnea_weight, "regular":1.0},
        args.dry,
    )

    run_result = run(
        RunCfg(
            experiment_name=experiment_name,
            model_name=args.model,
            model=model,
            trainer=trainer,
            train_datasets=[train_dataset],
            val_datasets=[] if val_dataset is None else [val_dataset],
            test_datasets=[("Ruhrland2024", test_dataset)],
            batch_size=BATCH_SIZE,
            n_samples=N_SAMPLES,
            num_workers_dataloader=NUM_WORKERS_DATALOADER,
            test_repeats=[1],
            use_energy_tracker=False,
            tags={"model": args.model},
            collate_fn=batch_collate,
            use_mlflow=True,
            log_path=os.path.join("results", "breathing"),
            meta_data=vars(args)
        )
    )

    export_prediction_package(
        os.path.join("results", "breathing", experiment_name, "deploy", experiment_name, ".swmodel"),
        model=run_result.model,
        trainer=run_result.trainer,
        dataset=train_dataset.to_unlabelled(),
        metadata={"experiment_name": experiment_name, **vars(args)},
    )


if __name__ == "__main__":
    main()
