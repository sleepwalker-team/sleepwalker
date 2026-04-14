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
from torch.utils.data import DataLoader, RandomSampler

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets import ChannelConfig, Ruhrlandklinik
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.normalizer.PulseFilterNormalizer import PulseFilterNormalizer
from sleepwalker.datasets.normalizer.RespirationFilterNormalizer import RespirationFilterNormalizer
from sleepwalker.datasets.normalizer.SaturationFilterNormalizer import SaturationFilterNormalizer
from sleepwalker.datasets.utils import export_batch_collate, get_edf_files_in_repo, random_split
from sleepwalker.models.UTime import UTime
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.utils.filtering import trim_wake
from sleepwalker.trainer.utils.splits import load_or_build_numpy_cache
from sleepwalker.trainer.utils.targets import prepare_multiclass_target
from sleepwalker.utils import logger, suppress_stdout_logging

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
mp.set_sharing_strategy("file_system")

TRAIN_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/train-test-2023"
TEST_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/val-2024"
EXPERIMENT_NAME = "desaturation"
BATCH_SIZE = 128
EPOCHS = 100
N_SAMPLES = 250_000
NUM_WORKERS_DATASET = 16
NUM_WORKERS_DATALOADER = 16
SAMPLE_FREQUENCY = 75
TOTAL_INPUT = "100s"
TARGET_RESOLUTION = "10s"
STRIDE = "2s"
SLEEP_PERCENTAGE = 0.5
TARGET_CLASSES = ["desaturation", "no desaturation"]
SLEEP_LABELS = ["n1", "n2", "n3", "rem"]
CHANNELS = ["Saturation", "SpO2 B-B", "Pulse Waveform", "PWA", "Pulse"]

EVENT_MAPPING = {
    "wach": "wake",
    "n1": "n1",
    "n2": "n2",
    "n3": "n3",
    "rem": "rem",
    "entsättigung": "desaturation",
}

prepare_desaturation_target = partial(
    prepare_multiclass_target,
    target_classes=TARGET_CLASSES,
    filters=[{"columns": SLEEP_LABELS, "percentage": SLEEP_PERCENTAGE, "mode": "min"}],
)

def build_channels():
    channels = [
        ChannelConfig(name="Chest", normalizer=RespirationFilterNormalizer(fs=SAMPLE_FREQUENCY), group=None),
        ChannelConfig(name="Abdomen", normalizer=RespirationFilterNormalizer(fs=SAMPLE_FREQUENCY), group=None),
        ChannelConfig(name="Saturation", normalizer=SaturationFilterNormalizer(fs=SAMPLE_FREQUENCY, clip_range=None), group=None),
        ChannelConfig(name="Pulse Waveform", normalizer=PulseFilterNormalizer(fs=SAMPLE_FREQUENCY), group=None),
    ]
    return channels

def prepare_patient(data_df, label_df, label_extra_df, patient=None):
    trimmed = trim_wake(data_df, label_df, label_extra_df)
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

def build_dataset(patients: list[str]):
    dataset = Ruhrlandklinik(
        channels=build_channels(),
        sample_frequency=SAMPLE_FREQUENCY,
        event_mapping=EVENT_MAPPING,
        stride=STRIDE,
        prepare_patient=prepare_patient,
        prepare_target=prepare_desaturation_target,
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

def load_split_dataset(
    patients: list[str],
    enable_cache: bool,
    n_samples: int | None = None,
    cache_path: str | None = None,
):
    if enable_cache:
        if cache_path is None:
            raise ValueError("cache_path must be provided when enable_cache is set.")
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

    return build_dataset(patients)

def build_model_and_trainer(train_dataset):
    model = UTime(
        ts_len=train_dataset.get_timeseries_len(),
        n_channels=len(train_dataset.get_input_channels()),
        classes=TARGET_CLASSES,
        sampling_frequency=SAMPLE_FREQUENCY,
        channel=[16, 32, 64, 128, 256],
        maxpool=[8, 6, 4, 4, 2],
        kernel=[5, 5, 5, 5, 5],
        norm="channel",
        mlp_size=64,
    )
    trainer = MulticlassTrainer(
        epochs=EPOCHS,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=75
        ),
        classes=TARGET_CLASSES,
        loss_function=torch.nn.functional.cross_entropy,
        save_every=10,
        loss_mode="inverse",
        early_stopping=5
    )
    return model, trainer


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a desaturation model.")
    parser.add_argument("--enable-cache", action="store_true", help="Load or build numpy-backed cached splits.")
    parser.add_argument("--cache-path", default=os.path.join("cache", "train_desaturation"), help="Base directory for train/test caches.")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of train patients reserved for validation. Set to 0 to disable.")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    train_cache_path = os.path.join(args.cache_path, "train")
    val_cache_path = os.path.join(args.cache_path, "val")
    test_cache_path = os.path.join(args.cache_path, "test")
    train_patients = [p for p in get_edf_files_in_repo(TRAIN_ROOT, recursive=True) if not is_pap_patient(p)]
    test_patients = [p for p in get_edf_files_in_repo(TEST_ROOT, recursive=True) if not is_pap_patient(p)]
    if args.dry:
        train_patients = train_patients[:2]
        test_patients = test_patients[:2]

    val_dataset = None
    if args.val_frac is not None and args.val_frac > 0:
        train_patients, val_patients = random_split(train_patients, test_frac=args.val_frac)
    else:
        val_patients = []

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_dataset = load_split_dataset(
            train_patients,
            args.enable_cache,
            1_000 if args.dry else N_SAMPLES,
            train_cache_path,
        )
        logger.uncontext()
        if len(val_patients) > 0:
            logger.context("VAL")
            val_dataset = load_split_dataset(
                val_patients,
                args.enable_cache,
                1_000 if args.dry else N_SAMPLES,
                val_cache_path,
            )
            logger.uncontext()
        logger.context("TEST")
        test_dataset = load_split_dataset(
            test_patients,
            args.enable_cache,
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
            val_datasets=[] if val_dataset is None else [val_dataset],
            test_datasets=[("test", test_dataset)],
            batch_size=BATCH_SIZE,
            n_samples=1_000 if args.dry else N_SAMPLES,
            num_workers_dataloader=NUM_WORKERS_DATALOADER,
            test_repeats=[1],
            use_energy_tracker=False,
            tags={"model": "UTime"},
            collate_fn=batch_collate,
            export_package_path=os.path.join("deployment","desaturation")
        )
    )


if __name__ == "__main__":
    main()
