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
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.normalizer.PulseFilterNormalizer import PulseFilterNormalizer
from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer
from sleepwalker.datasets.utils import get_edf_files_in_repo, random_split
from sleepwalker.models import MultiModel, MetaModelEntry, SleepTransformer
from sleepwalker.models.UTime import UTime
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.utils.filtering import trim_wake
from sleepwalker.trainer.utils.targets import prepare_multiclass_target
from sleepwalker.utils import logger, suppress_stdout_logging

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
mp.set_sharing_strategy("file_system")

TRAIN_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/train-test-2023"
TEST_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/val-2024"
EXPERIMENT_NAME = "arousal"
BATCH_SIZE = 128
EPOCHS = 100
N_SAMPLES = 250_000
NUM_WORKERS_DATASET = 16
NUM_WORKERS_DATALOADER = 16
SAMPLE_FREQUENCY = 100
TOTAL_INPUT = "10s"
TARGET_RESOLUTION = "10s"
STRIDE = "1s"
SLEEP_PERCENTAGE = 0.5
IMPEDANCE_CUTOFF_OHM = 20_000.0
TARGET_CLASSES = ["no_arousal", "arousal"]
SLEEP_LABELS = ["n1", "n2", "n3", "rem"]
EEG_GROUP_CHANNELS = ["C3-M2", "C4-M1", "E1-M2", "E2-M1"]
EEG_QUALITY_CHANNELS = {
    "C3-M2": "C3 Impedanz",
    "C4-M1": "C4 Impedanz",
    "E1-M2": "E1 Impedanz",
    "E2-M1": "E2 Impedanz",
}

EVENT_MAPPING = {
    "wach": "wake",
    "n1": "n1",
    "n2": "n2",
    "n3": "n3",
    "rem": "rem",
    "arousal": "arousal",
    "artefakt": "artifact",
    "rera": "arousal",
}

prepare_arousal_target = partial(
    prepare_multiclass_target,
    target_classes=TARGET_CLASSES,
    filters=[
        {"columns": SLEEP_LABELS, "percentage": SLEEP_PERCENTAGE, "mode": "min"},
        {"columns": ["wake"], "percentage": 0.5, "mode": "max"},
        {"columns": ["artifact"], "percentage": 0.0, "mode": "max"},
    ],
)


def build_channels(grouped: bool):
    channels = []
    if grouped:
        channels.extend(
            [
                ChannelConfig(
                    name=channel,
                    normalizer=EEGFilterNormalizer(fs=SAMPLE_FREQUENCY),
                    group="EEG",
                    quality_name=EEG_QUALITY_CHANNELS[channel],
                )
                for channel in EEG_GROUP_CHANNELS
            ]
        )
    else:
        channels.append(
            ChannelConfig(
                name="C3-M2",
                normalizer=EEGFilterNormalizer(fs=SAMPLE_FREQUENCY),
            )
        )
    channels.extend(
        [
            ChannelConfig(
                name="ECG",
                normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, lowcut=0.3, highcut=35.0, notch_freq=50.0),
                group=None,
            ),
            ChannelConfig(name="Pulse Waveform", normalizer=PulseFilterNormalizer(fs=SAMPLE_FREQUENCY), group=None),
        ]
    )
    return channels


def prepare_patient(data_df, label_df, label_extra_df, patient=None):
    trimmed = trim_wake(data_df, label_df, label_extra_df)
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df


def prepare_sample(
    data,
    quality_data=None,
    target=None,
    target_extra=None,
    patient=None,
    time=None,
    impedance_cutoff_ohm: float = IMPEDANCE_CUTOFF_OHM,
):
    if quality_data is not None and "EEG" in quality_data.columns:
        if float(quality_data["EEG"].astype(float).mean()) > impedance_cutoff_ohm:
            return None

    if float(data.isna().mean().mean()) > 0.05:
        return None

    for col in ["EEG", "C3-M2", "ECG", "Pulse Waveform"]:
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

def build_dataset(patients: list[str], grouped: bool, impedance_cutoff_ohm: float):
    dataset = Ruhrlandklinik(
        channels=build_channels(grouped),
        sample_frequency=SAMPLE_FREQUENCY,
        event_mapping=EVENT_MAPPING,
        stride=STRIDE,
        prepare_patient=prepare_patient,
        prepare_target=prepare_arousal_target,
        prepare_sample=partial(prepare_sample, impedance_cutoff_ohm=impedance_cutoff_ohm),
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


def has_required_channels(edf_path: str, grouped: bool) -> bool:
    meta = read_edf_meta(edf_path)
    available_channels = set(meta["signals"])
    if grouped:
        has_eeg = any(channel in available_channels for channel in EEG_GROUP_CHANNELS)
    else:
        has_eeg = "C3-M2" in available_channels
    return has_eeg and "ECG" in available_channels and "Pulse Waveform" in available_channels


def list_patients(source_root: str, grouped: bool, dry_run: bool) -> list[str]:
    patients = [
        patient
        for patient in get_edf_files_in_repo(source_root, recursive=True)
        if not is_pap_patient(patient) and has_required_channels(patient, grouped)
    ]
    return patients[:2] if dry_run else patients


def load_split_dataset(
    patients: list[str],
    grouped: bool,
    impedance_cutoff_ohm: float,
):
    return build_dataset(patients, grouped, impedance_cutoff_ohm)


def build_model_and_trainer(train_dataset):
    eeg_channels = [channel for channel in ["EEG", "C3-M2"] if channel in train_dataset.get_input_channels()]
    aux_channels = [channel for channel in train_dataset.get_input_channels() if channel not in eeg_channels]
    if len(eeg_channels) != 1:
        raise ValueError(f"Arousal MultiModel expects exactly one EEG input channel, found {eeg_channels}.")
    if len(aux_channels) == 0:
        raise ValueError("Arousal MultiModel requires at least one non-EEG input channel.")

    aux_model = UTime(
        ts_len=train_dataset.get_timeseries_len(),
        n_channels=len(aux_channels),
        classes=None,
        sampling_frequency=SAMPLE_FREQUENCY,
        channel=[16, 32, 64, 128],
        maxpool=[10, 8, 6, 4],
        kernel=[5, 5, 5, 5],
        norm="channel",
        mlp_size=64,
    )
    eeg_model = SleepTransformer(
        classes=None,
        n_channels=1,
    )
    model = MultiModel(
        classes=TARGET_CLASSES,
        input_channels=train_dataset.get_input_channels(),
        models=[
            MetaModelEntry(aux_model, aux_channels),
            MetaModelEntry(eeg_model, eeg_channels),
        ],
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
    parser = argparse.ArgumentParser(description="Train and evaluate an arousal model.")
    parser.add_argument("--grouped", action="store_true", help="Randomly sample one available EEG/EOG channel into a grouped EEG input.")
    parser.add_argument(
        "--impedance-cutoff-ohm",
        type=float,
        default=IMPEDANCE_CUTOFF_OHM,
        help="Reject grouped EEG windows whose mean impedance exceeds this threshold.",
    )
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of train patients reserved for validation. Set to 0 to disable.")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    train_patients = list_patients(TRAIN_ROOT, args.grouped, args.dry)
    test_patients = list_patients(TEST_ROOT, args.grouped, args.dry)

    val_dataset = None
    if args.val_frac is not None and args.val_frac > 0:
        train_patients, val_patients = random_split(train_patients, test_frac=args.val_frac)
    else:
        val_patients = []

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_dataset = load_split_dataset(
            train_patients,
            args.grouped,
            args.impedance_cutoff_ohm,
        )
        logger.uncontext()
        if len(val_patients) > 0:
            logger.context("VAL")
            val_dataset = load_split_dataset(
                val_patients,
                args.grouped,
                args.impedance_cutoff_ohm,
            )
            logger.uncontext()
        logger.context("TEST")
        test_dataset = load_split_dataset(
            test_patients,
            args.grouped,
            args.impedance_cutoff_ohm,
        )
        logger.uncontext()

    model, trainer = build_model_and_trainer(train_dataset)
    experiment_name = f"{EXPERIMENT_NAME}-grouped" if args.grouped else EXPERIMENT_NAME
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
        )
    )


if __name__ == "__main__":
    main()
