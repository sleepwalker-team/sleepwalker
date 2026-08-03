#!/bin/env python3

from __future__ import annotations

import argparse
import os
from functools import partial

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import torch
import torch.multiprocessing as mp

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets import ChannelConfig, Ruhrlandklinik
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.normalizer.PulseFilterNormalizer import PulseFilterNormalizer
from sleepwalker.datasets.normalizer.RespirationFilterNormalizer import RespirationFilterNormalizer
from sleepwalker.datasets.normalizer.SaturationFilterNormalizer import SaturationFilterNormalizer
from sleepwalker.datasets.utils import get_edf_files_in_repo, random_split
from sleepwalker.models.UTime import UTime
from sleepwalker.models.preprocessors.RobustScaler import RobustScaler
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run, seed_everything
from sleepwalker.trainer.utils.filtering import trim_event
from sleepwalker.trainer.utils.targets import prepare_multiclass_target
from sleepwalker.utils import logger, suppress_stdout_logging

try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
mp.set_sharing_strategy("file_system")

TRAIN_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/train-test-2023"
TEST_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/val-2024"
EXPERIMENT_NAME = "desaturation"
NUM_WORKERS_DATASET = 16
NUM_WORKERS_DATALOADER = 16
SLEEP_PERCENTAGE = 0.5
TARGET_CLASSES = ["desaturation", "no desaturation"]

EVENT_MAPPING = {
    "wach": "wake",
    "n1": "sleep",
    "n2": "sleep",
    "n3": "sleep",
    "rem": "sleep",
    "entsättigung": "desaturation",
}

def build_channel_config(channel_name: str, sample_frequency: int) -> ChannelConfig:
    if channel_name == "Chest":
        return ChannelConfig(name="Chest", normalizer=RespirationFilterNormalizer(fs=sample_frequency), group=None, unit="V")
    if channel_name == "Abdomen":
        return ChannelConfig(name="Abdomen", normalizer=RespirationFilterNormalizer(fs=sample_frequency), group=None, unit="V")
    if channel_name == "Saturation":
        return ChannelConfig(
            name="Saturation",
            normalizer=SaturationFilterNormalizer(fs=sample_frequency, clip_range=None),
            group=None,
            unit="%",
        )
    if channel_name == "Pulse Waveform":
        return ChannelConfig(name="Pulse Waveform", normalizer=PulseFilterNormalizer(fs=sample_frequency), group=None)
    raise ValueError(f"Did not recognize channel {channel_name}")


def build_channels(channel_names: list[str], sample_frequency: int) -> list[ChannelConfig]:
    return [build_channel_config(channel_name, sample_frequency) for channel_name in channel_names]

def prepare_patient(data_df, label_df, label_extra_df, patient=None):
    trimmed = trim_event(data_df, label_df, label_extra_df, ["sleep"])
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df

def prepare_sample(data, quality_data=None, target=None, target_extra=None, patient=None, time=None):
    if float(data.isna().mean().mean()) > 0.05:
        return None

    if "Saturation" in data.columns:
        invalid_spo2_fraction = float(((data["Saturation"] < -5.0) | (data["Saturation"] > 5.0)).mean())
        if invalid_spo2_fraction > 0.05:
            return None

    for col in ["EEG", "Chest", "Abdomen", "Pulse Waveform"]:
        if col in data.columns and float(data[col].std()) < 1e-3:
            return None

    item = {
        "data": torch.from_numpy(data.values).float(),
        "target": target,
        "patient": patient,
        "time": time,
    }
    if target_extra is not None:
        item["target_extra"] = target_extra
    return item

def build_dataset_template(
    channels: list[str],
    sample_frequency: int,
    stride: str,
    total_input: str,
    target_resolution: str,
    sleep_percentage: float,
):
    prepare_desaturation_target = partial(
        prepare_multiclass_target,
        target_classes=TARGET_CLASSES,
        filters=[{"columns": ["sleep"], "percentage": sleep_percentage, "mode": "min"}],
    )
    
    dataset = Ruhrlandklinik(
        channels=build_channels(channels, sample_frequency),
        sample_frequency=sample_frequency,
        event_mapping=EVENT_MAPPING,
        stride=stride,
        prepare_patient=prepare_patient,
        prepare_target=prepare_desaturation_target,
        prepare_sample=prepare_sample,
        total_input=total_input,
        target_resolution=target_resolution,
    )
    dataset.classes = list(TARGET_CLASSES)
    return dataset


def build_dataset(
    patients: list[str],
    channels: list[str],
    sample_frequency: int,
    stride: str,
    total_input: str,
    target_resolution: str,
    sleep_percentage: float,
    num_workers_dataset: int = NUM_WORKERS_DATASET,
):
    dataset = build_dataset_template(
        channels=channels,
        sample_frequency=sample_frequency,
        stride=stride,
        total_input=total_input,
        target_resolution=target_resolution,
        sleep_percentage=sleep_percentage,
    )
    dataset.initialize(patients, num_workers_dataset)
    return dataset

def is_pap_patient(edf_path: str) -> bool:
    meta = read_edf_meta(edf_path)
    available_channels = meta["signals"]
    PAP_CHANNEL_PATTERNS = [
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
    return any(ch in available_channels for ch in PAP_CHANNEL_PATTERNS)


def has_required_channels(edf_path: str, channels: list[str]) -> bool:
    meta = read_edf_meta(edf_path)
    available_channels = set(meta["signals"])
    return all(channel in available_channels for channel in channels)


def list_patients(source_root: str, channels: list[str], dry_run: bool) -> list[str]:
    patients = [
        patient
        for patient in get_edf_files_in_repo(source_root, recursive=True)
        if not is_pap_patient(patient) and has_required_channels(patient, channels)
    ]
    return patients[:2] if dry_run else patients


def build_model_and_trainer(
    train_dataset,
    model_name: str,
    sample_frequency: int,
    epochs: int,
    scaler: bool,
    desaturation_weight: int,
):
    n_channels = len(train_dataset.get_input_channels())
    preprocessors = [RobustScaler(lower_quantile=0.1, upper_quantile=0.9, channels=[i for i in range(n_channels)])] if scaler else None

    if model_name == "utime-big":
        model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=n_channels,
            classes=TARGET_CLASSES,
            sampling_frequency=sample_frequency,
            channel=[16, 32, 64, 128, 256],
            maxpool=[8, 6, 4, 4, 2],
            kernel=[5, 5, 5, 5, 5],
            norm="channel",
            mlp_size=256,
            preprocessors=preprocessors,
        )
    elif model_name == "utime-small":
        model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=n_channels,
            classes=TARGET_CLASSES,
            sampling_frequency=sample_frequency,
            channel=[16, 32, 64],
            maxpool=[8, 6, 4],
            kernel=[5, 5, 5],
            norm="channel",
            mlp_size=64,
            preprocessors=preprocessors,
        )
    else:
        raise ValueError(f"Did not recognize model {model_name}")
    trainer = MulticlassTrainer(
        epochs=epochs,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ),
        classes=TARGET_CLASSES,
        loss_function=torch.nn.functional.cross_entropy,
        save_every=10,
        loss_mode="inverse",
        early_stopping=5,
        class_weights={"no desaturation": 1, "desaturation": desaturation_weight},
    )
    return model, trainer


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a desaturation model.")
    parser.add_argument("--id", type=str, default="", help="ID of the experiment.")
    parser.add_argument("--channels", nargs='+', default=["Chest", "Abdomen", "Saturation", "Pulse Waveform"], help="List of channels.")
    parser.add_argument("--scaler", action="store_true", help="If RobustScaler should be used.")
    parser.add_argument("--desaturation_weight", type=int, default=1, help="Weight of desaturation class.")
    parser.add_argument("--model", type=str, default="utime-big", help="What model to use.")
    parser.add_argument("--sample_frequency", type=int, default=75, help="Input sample frequency in Hz.")
    parser.add_argument("--total_input", type=str, default="100s", help="Total input size.")
    parser.add_argument("--target_resolution", type=str, default="10s", help="Target resolution.")
    parser.add_argument("--sleep_percentage", type=float, default=0.5, help="Sleep percentage for filtering.")
    parser.add_argument("--stride", type=str, default="2s", help="Stride between samples.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--n_samples", type=int, default=250_000, help="Number of training samples per epoch.")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Fraction of train patients reserved for validation. Set to 0 to disable.")
    parser.add_argument("--max_train_patients", type=int, default=None, help="Optional cap for training patients after listing.")
    parser.add_argument("--max_test_patients", type=int, default=None, help="Optional cap for test patients after listing.")
    parser.add_argument("--num_workers_dataset", type=int, default=NUM_WORKERS_DATASET, help="Workers for patient initialization.")
    parser.add_argument("--num_workers_dataloader", type=int, default=NUM_WORKERS_DATALOADER, help="Workers for torch dataloaders.")
    parser.add_argument("--no_mlflow", action="store_true", help="Disable MLflow logging for smoke tests.")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    seed_everything(args.seed)

    experiment_name = f"{EXPERIMENT_NAME}_{args.id}"
    if args.dry:
        logger.info("Performing dry run to test pipeline!")
        experiment_name += "-dev"

    os.makedirs(os.path.join("results", "desaturation", experiment_name), exist_ok=True)
    logger.set_log_file(os.path.join("results", "desaturation", experiment_name, "output.log"))

    train_patients = list_patients(TRAIN_ROOT, args.channels, args.dry)
    test_patients = list_patients(TEST_ROOT, args.channels, args.dry)
    if args.max_train_patients is not None:
        train_patients = train_patients[: args.max_train_patients]
    if args.max_test_patients is not None:
        test_patients = test_patients[: args.max_test_patients]

    val_dataset = None
    if args.val_frac is not None and args.val_frac > 0:
        train_patients, val_patients = random_split(train_patients, test_frac=args.val_frac, seed=args.seed)
    else:
        val_patients = []

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_dataset = build_dataset(
            train_patients,
            args.channels,
            args.sample_frequency,
            args.stride,
            args.total_input,
            args.target_resolution,
            args.sleep_percentage,
            args.num_workers_dataset,
        )
        logger.uncontext()
        if len(val_patients) > 0:
            logger.context("VAL")
            val_dataset = build_dataset(
                val_patients,
                args.channels,
                args.sample_frequency,
                args.stride,
                args.total_input,
                args.target_resolution,
                args.sleep_percentage,
                args.num_workers_dataset,
            )
            logger.uncontext()
        logger.context("TEST")
        test_dataset = build_dataset(
            test_patients,
            args.channels,
            args.sample_frequency,
            args.stride,
            args.total_input,
            args.target_resolution,
            args.sleep_percentage,
            args.num_workers_dataset,
        )
        logger.uncontext()

    model, trainer = build_model_and_trainer(
        train_dataset,
        args.model,
        args.sample_frequency,
        2 if args.dry else args.epochs,
        args.scaler,
        args.desaturation_weight,
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
            batch_size=args.batch_size,
            n_samples=1_000 if args.dry else args.n_samples,
            num_workers_dataloader=args.num_workers_dataloader,
            n_samples_test=1_000 if args.dry else None,
            test_repeats=[1],
            tags={"model": args.model},
            collate_fn=batch_collate,
            use_mlflow=not args.no_mlflow,
            log_path=os.path.join("results", "desaturation"),
            meta_data=vars(args),
            expert_task="desaturation",
        )
    )


if __name__ == "__main__":
    main()
