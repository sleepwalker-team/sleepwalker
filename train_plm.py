#!/bin/env python3

from __future__ import annotations

import argparse
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, RandomSampler

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets import ChannelConfig, Ruhrlandklinik
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer
from sleepwalker.datasets.utils import export_batch_collate, get_edf_files_in_repo
from sleepwalker.models.UTime import UTime
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.utils.filtering import trim_wake
from sleepwalker.trainer.utils.splits import load_or_build_numpy_cache
from sleepwalker.utils import logger, suppress_stdout_logging

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
mp.set_sharing_strategy("file_system")

TRAIN_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/train-test-2023"
TEST_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/val-2024"
EXPERIMENT_NAME = "plm"
BATCH_SIZE = 128
EPOCHS = 100
N_SAMPLES = 250_000
NUM_WORKERS_DATASET = 16
NUM_WORKERS_DATALOADER = 16
SAMPLE_FREQUENCY = 75
TOTAL_INPUT = "10s"
TARGET_RESOLUTION = "10s"
STRIDE = "1s"
SLEEP_PERCENTAGE = 0.5
TARGET_CLASSES = ["no_plm", "plm"]
SLEEP_LABELS = ["n1", "n2", "n3", "rem"]

EVENT_MAPPING = {
    "wach": "wake",
    "n1": "n1",
    "n2": "n2",
    "n3": "n3",
    "rem": "rem",
    "lm": "plm",
    "plm": "plm",
    "plms": "plm",
    "plm-arousal": "plm",
    "artefakt": "artifact",
    "bewegung": "movement",
}


def build_channels():
    return [
        ChannelConfig(
            name="Left Leg",
            normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, lowcut=5.0, highcut=30.0, notch_freq=50.0),
            group=None,
        ),
        ChannelConfig(
            name="Right Leg",
            normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, lowcut=5.0, highcut=30.0, notch_freq=50.0),
            group=None,
        ),
        ChannelConfig(name="Activity", normalizer=None, group=None),
    ]


def prepare_patient(data_df, label_df, label_extra_df, patient=None):
    trimmed = trim_wake(data_df, label_df, label_extra_df)
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df


def prepare_sample(data, quality_data=None, target=None, target_extra=None, patient=None, time=None):
    if float(data.isna().mean().mean()) > 0.05:
        return None

    for col in ["Left Leg", "Right Leg"]:
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


def _merge_target_frames(target, target_extra, columns):
    current = target.reindex(columns=columns, fill_value=0) if target is not None else None
    extra = target_extra.reindex(columns=columns, fill_value=0) if target_extra is not None else None
    if current is None and extra is None:
        return None
    if current is None:
        return extra
    if extra is None:
        return current
    return current.add(extra, fill_value=0)


def prepare_plm_target(target, target_extra=None, patient=None, time=None, percentage: float = 0.5):
    if target is None:
        return None

    sleep_target = _merge_target_frames(target, target_extra, SLEEP_LABELS)
    if sleep_target is not None:
        sleep_fraction = float(sleep_target.any(axis=1).mean())
        if sleep_fraction < SLEEP_PERCENTAGE:
            return None

    ignore_target = _merge_target_frames(target, target_extra, ["artifact", "movement"])
    if ignore_target is not None:
        if float(ignore_target.any(axis=1).mean()) > 0.0:
            return None

    def _build_onehot(current_target, current_target_extra):
        merged = _merge_target_frames(current_target, current_target_extra, ["plm"])
        if merged is None:
            return None
        freq = pd.to_timedelta(merged.index.freq).total_seconds()
        threshold = len(merged) * freq * percentage
        positive_seconds = float(merged["plm"].sum() * freq)
        onehot = torch.zeros(2, dtype=torch.float32)
        onehot[1 if positive_seconds >= threshold else 0] = 1.0
        return onehot

    target_onehot = _build_onehot(target, target_extra)
    if target_onehot is None:
        return None

    item = {"target": target_onehot}
    if target_extra is not None:
        target_extra_onehot = _build_onehot(target_extra, None)
        if target_extra_onehot is not None:
            item["target_extra"] = target_extra_onehot
    return item


def build_dataset(patients: list[str]):
    dataset = Ruhrlandklinik(
        channels=build_channels(),
        sample_frequency=SAMPLE_FREQUENCY,
        event_mapping=EVENT_MAPPING,
        stride=STRIDE,
        prepare_patient=prepare_patient,
        prepare_target=prepare_plm_target,
        prepare_sample=prepare_sample,
        total_input=TOTAL_INPUT,
        target_resolution=TARGET_RESOLUTION,
    )
    dataset.classes = list(TARGET_CLASSES)
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


def has_required_channels(edf_path: str) -> bool:
    meta = read_edf_meta(edf_path)
    available_channels = set(meta["signals"])
    return any(channel in available_channels for channel in ["Left Leg", "Right Leg"])


def list_patients(source_root: str, dry_run: bool) -> list[str]:
    patients = [
        patient
        for patient in get_edf_files_in_repo(source_root, recursive=True)
        if not is_pap_patient(patient) and has_required_channels(patient)
    ]
    return patients[:2] if dry_run else patients


def load_split_dataset(
    source_root: str,
    enable_cache: bool,
    dry_run: bool,
    n_samples: int | None = None,
    cache_path: str | None = None,
):
    if enable_cache:
        if cache_path is None:
            raise ValueError("cache_path must be provided when enable_cache is set.")
        patients = list_patients(source_root, dry_run)
        dataset = build_dataset(patients)
        sampler = RandomSampler(dataset, num_samples=n_samples) if n_samples is not None else None
        export_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            sampler=sampler,
            num_workers=NUM_WORKERS_DATALOADER,
            collate_fn=export_batch_collate,
            drop_last=False,
            persistent_workers=NUM_WORKERS_DATALOADER > 0,
        )
        return load_or_build_numpy_cache(
            export_loader,
            cache_path,
            in_memory=False,
        )

    patients = list_patients(source_root, dry_run)
    return build_dataset(patients)


def build_model_and_trainer(train_dataset):
    model = UTime(
        ts_len=train_dataset.get_timeseries_len(),
        n_channels=len(train_dataset.get_input_channels()),
        classes=TARGET_CLASSES,
        sampling_frequency=SAMPLE_FREQUENCY,
        channel=[16, 32, 64, 128],
        maxpool=[10, 8, 6, 4],
        kernel=[5, 5, 3, 3],
        norm="channel",
        mlp_size=64,
    )
    trainer = MulticlassTrainer(
        epochs=EPOCHS,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ),
        classes=TARGET_CLASSES,
        loss_function=torch.nn.functional.cross_entropy,
        save_every=10,
        loss_mode="inverse",
    )
    return model, trainer


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a PLM model.")
    parser.add_argument("--enable-cache", action="store_true", help="Load or build numpy-backed cached splits.")
    parser.add_argument("--cache-path", default=os.path.join("cache", "train_plm"), help="Base directory for train/test caches.")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    train_cache_path = os.path.join(args.cache_path, "train")
    test_cache_path = os.path.join(args.cache_path, "test")

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_dataset = load_split_dataset(
            TRAIN_ROOT,
            args.enable_cache,
            args.dry,
            1_000 if args.dry else N_SAMPLES,
            train_cache_path,
        )
        logger.uncontext()
        logger.context("TEST")
        test_dataset = load_split_dataset(
            TEST_ROOT,
            args.enable_cache,
            args.dry,
            None,
            test_cache_path,
        )
        logger.uncontext()

    model, trainer = build_model_and_trainer(train_dataset)
    experiment_name = f"{EXPERIMENT_NAME}"
    if args.dry:
        logger.info("Performing dry run to test pipeline!")
        experiment_name += "-dev"

    run(
        RunCfg(
            experiment_name=experiment_name,
            model_name="UTime",
            model=model,
            trainer=trainer,
            train_datasets=[train_dataset],
            val_datasets=[],
            test_datasets=[("test", test_dataset)],
            batch_size=BATCH_SIZE,
            n_samples=1_000 if args.dry else N_SAMPLES,
            num_workers_dataloader=NUM_WORKERS_DATALOADER,
            test_repeats=[1],
            use_energy_tracker=False,
            tags={"model": "UTime"},
            collate_fn=batch_collate,
        )
    )


if __name__ == "__main__":
    main()
