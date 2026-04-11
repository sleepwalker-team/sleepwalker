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
from sleepwalker.datasets.utils import export_batch_collate, get_edf_files_in_repo
from sleepwalker.models.UTime import UTime
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.utils.splits import load_or_build_numpy_cache
from sleepwalker.utils import logger, suppress_stdout_logging

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
mp.set_sharing_strategy("file_system")

TRAIN_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/train-test-2023"
TEST_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/val-2024"
EXPERIMENT_NAME = "body_position"
BATCH_SIZE = 128
EPOCHS = 100
N_SAMPLES = 250_000
NUM_WORKERS_DATASET = 16
NUM_WORKERS_DATALOADER = 16
SAMPLE_FREQUENCY = 75
TOTAL_INPUT = "30s"
TARGET_RESOLUTION = "30s"
STRIDE = "5s"
TARGET_CLASSES = ["supine", "left", "right", "prone", "upright"]

EVENT_MAPPING = {
    "rückenlage": "supine",
    "links": "left",
    "rechts": "right",
    "bauchlage": "prone",
    "aufrecht": "upright",
    "unbekannt": "unknown",
    "bewegung": "movement",
}


def build_channels():
    return [
        ChannelConfig(name="X Axis", normalizer=None, group=None),
        ChannelConfig(name="Y Axis", normalizer=None, group=None),
        ChannelConfig(name="Z Axis", normalizer=None, group=None),
        ChannelConfig(name="Activity", normalizer=None, group=None),
    ]


def prepare_sample(data, quality_data=None, target=None, target_extra=None, patient=None, time=None):
    if float(data.isna().mean().mean()) > 0.10:
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


def prepare_body_position_target(target, target_extra=None, patient=None, time=None, percentage: float = 0.5):
    if target is None:
        return None

    ignore_target = _merge_target_frames(target, target_extra, ["unknown", "movement"])
    if ignore_target is not None and float(ignore_target.any(axis=1).mean()) > 0.0:
        return None

    position_target = _merge_target_frames(target, target_extra, TARGET_CLASSES)
    if position_target is None:
        return None

    freq = pd.to_timedelta(position_target.index.freq).total_seconds()
    threshold = len(position_target) * freq * percentage
    durations = torch.tensor(position_target.sum().to_numpy() * freq, dtype=torch.float32)
    active = durations >= threshold
    if int(active.sum().item()) != 1:
        return None

    onehot = torch.zeros(len(TARGET_CLASSES), dtype=torch.float32)
    onehot[int(active.nonzero(as_tuple=False).item())] = 1.0

    item = {"target": onehot}
    if target_extra is not None:
        item["target_extra"] = onehot.clone()
    return item


def build_dataset(patients: list[str]):
    dataset = Ruhrlandklinik(
        channels=build_channels(),
        sample_frequency=SAMPLE_FREQUENCY,
        event_mapping=EVENT_MAPPING,
        stride=STRIDE,
        prepare_patient=None,
        prepare_target=prepare_body_position_target,
        prepare_sample=prepare_sample,
        total_input=TOTAL_INPUT,
        target_resolution=TARGET_RESOLUTION,
    )
    dataset.classes = list(TARGET_CLASSES)
    dataset.initialize(patients, NUM_WORKERS_DATASET)
    return dataset


def has_required_channels(edf_path: str) -> bool:
    meta = read_edf_meta(edf_path)
    available_channels = set(meta["signals"])
    return any(channel in available_channels for channel in ["PosAngle", "X Axis", "Y Axis", "Z Axis", "Elevation", "Activity"])


def list_patients(source_root: str, dry_run: bool) -> list[str]:
    patients = [patient for patient in get_edf_files_in_repo(source_root, recursive=True) if has_required_channels(patient)]
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
    parser = argparse.ArgumentParser(description="Train and evaluate a body-position model.")
    parser.add_argument("--enable-cache", action="store_true", help="Load or build numpy-backed cached splits.")
    parser.add_argument("--cache-path", default=os.path.join("cache", "train_body_position"), help="Base directory for train/test caches.")
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
