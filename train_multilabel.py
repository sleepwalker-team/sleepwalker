"""Ruhrland multitask training script for the current lab workflow.

This script assembles a Ruhrland-specific multitask dataset, quality filters,
`MetaModel`, and `MultiLabelTrainer` into an internal multitask experiment. The
configuration is tightly coupled to the current lab environment and should not
be treated as a stable public CLI.
"""

#!/bin/env python3

import argparse
from functools import partial
import os

import numpy as np
import pandas as pd
import torch

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets import ChannelConfig, Ruhrlandklinik
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.normalizer.PulseFilterNormalizer import PulseFilterNormalizer
from sleepwalker.datasets.normalizer.RespirationFilterNormalizer import RespirationFilterNormalizer
from sleepwalker.datasets.normalizer.SaturationFilterNormalizer import SaturationFilterNormalizer
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.utils import get_edf_files_in_repo
from sleepwalker.models import MetaModel, MetaModelEntry, SleepTransformer
from sleepwalker.models.UTime import UTime
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.trainer.utils.filtering import trim_event
from sleepwalker.utils import MlflowSink, logger

# TODO METAMODEL
#   ADD A POST-EMBEDDING MODEL (NOT FC)
#   ALLOW MULTIPLE INPUT RESOLUTIONS?!?!

edf_folder = "/raid/sleepwalker/ruhrlandklinik/raw/train-test-2023"
edf_folder_test = "/raid/sleepwalker/ruhrlandklinik/raw/val-2024"
batch_size = 128
epochs = 100
sample_frequency = 100
total_input = "630s"
target_resolution = "30s"
n_samples = None
experiment_name = "ruhrland_metamodel_multilabel"
num_workers_dataset = 8
num_workers_dataloader = 8
patient_filter_quantile = 0.05
impedance_cutoff_ohm = 20_000.0

respiratory_channels = ["Chest", "Abdomen", "Saturation", "Pulse Waveform"]
sleep_channels = ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"]
impedance_channels = {
    "F3-M2": "F3 Impedanz",
    "F4-M1": "F4 Impedanz",
    "C3-M2": "C3 Impedanz",
    "C4-M1": "C4 Impedanz",
    "O1-M2": "O1 Impedanz",
    "O2-M1": "O2 Impedanz",
}

channels_cfgs = [
    ChannelConfig(
        name=c,
        normalizer=EEGFilterNormalizer(fs=sample_frequency),
        group="EEG",
        quality_name=impedance_channels[c],
    )
    for c in sleep_channels
] + [
    ChannelConfig(name="Chest", normalizer=RespirationFilterNormalizer(fs=sample_frequency), group=None),
    ChannelConfig(name="Abdomen", normalizer=RespirationFilterNormalizer(fs=sample_frequency), group=None),
    ChannelConfig(name="Saturation", normalizer=SaturationFilterNormalizer(fs=sample_frequency,clip_range=None), group=None),
    ChannelConfig(name="Pulse Waveform", normalizer=PulseFilterNormalizer(fs=sample_frequency), group=None),
]

def has_required_channels(edf_path):
    """Check whether an EDF file exposes the channels required by this script."""
    meta = read_edf_meta(edf_path)
    return all(c in meta["signals"] for c in sleep_channels) and all(
        channel in meta["signals"] for channel in respiratory_channels + list(impedance_channels.values())
    )

event_mapping = {
    "wach": "wake",
    "n1": "n1",
    "n2": "n2",
    "n3": "n3",
    "rem": "rem",
    "a. gemischt": "obstruction",
    "a. obstruktiv": "obstruction",
    "a. zentral": "obstruction",
    "apnoe": "obstruction",
    "cheyne stokes": "obstruction",
    "fg apnoe geschlossen": "obstruction",
    "fg apnoe geöffnet": "obstruction",
    "fg-apnoe unbekannt": "obstruction",
    "h. obstruktiv": "obstruction",
    "h. zentral": "obstruction",
    "fg hypopnoe": "obstruction",
    "hypopnea-gemischt": "obstruction",
    "hypopnoe": "obstruction",
    "arousal": "arousal",
    "plm-arousal": "arousal",
    "rera": "arousal",
    "entsättigung": "desaturation",
}

task_config = {
    "breathing": {
        "labels": ["obstruction", "regular"],
        "default": "regular",
        "percentage": 0.5,
        "target_resolution": "10s",
        "loss_function": torch.nn.functional.cross_entropy,
        "loss_mode": "inverse",
    },
    "arousal": {
        "labels": ["arousal", "no arousal"],
        "default": "no arousal",
        "percentage": 0.5,
        "target_resolution": "1s",
        "loss_function": torch.nn.functional.cross_entropy,
        "loss_mode": "inverse",
    },
    "desat": {
        "labels": ["desaturation", "no desaturation"],
        "default": "no desaturation",
        "percentage": 0.5,
        "target_resolution": "10s",
        "loss_function": torch.nn.functional.cross_entropy,
        "loss_mode": "inverse",
    },
    "sleep": {
        "labels": ["n1", "n2", "n3", "rem", "wake"],
        "default": None,
        "percentage": 0.5,
        "target_resolution": "30s",
        "loss_function": torch.nn.functional.cross_entropy,
        "loss_mode": "inverse",
    },
}
normalized_task_config = MultiLabelTrainer.normalize_task_config(task_config)

def prepare_sleep_staging_patient(data_df, label_df, label_extra_df, patient=None):
    """Trim leading and trailing wake before multitask target extraction."""
    trimmed = trim_event(data_df, label_df, label_extra_df)
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df


def prepare_multilabel_sample(data, quality_data=None, target=None, target_extra=None, patient=None, time=None):
    """Apply signal-quality rejection and build one multitask sample item.

    Returns:
        A sample dictionary or `None` when the window fails the current
        quality heuristics.

    Notes:
        The exact thresholds are experiment-specific and are documented here as
        current script behavior, not as validated general defaults.
    """
    # Reject high-impedance windows before they reach the
    # model while keeping impedance out of the actual model inputs.
    if quality_data is not None and "EEG" in quality_data.columns:
        if float(quality_data["EEG"].astype(float).mean()) > impedance_cutoff_ohm:
            return None

    # Reject windows with too many missing values. We do this before channel-
    # specific checks so obviously corrupted windows exit quickly.
    if float(data.isna().mean().mean()) > 0.05:
        return None

    # Reject windows whose saturation signal contains many impossible values.
    # A small fraction of outliers can happen, but sustained invalid values
    # usually indicate sensor failure or export issues.
    if "Saturation" in data.columns:
        invalid_spo2_fraction = float(((data["Saturation"] < -5.0) | (data["Saturation"] > 5.0)).mean())
        if invalid_spo2_fraction > 0.05:
            return None

    # Reject near-flat windows in EEG and respiratory channels. The normalizers
    # already filter and z-normalize per patient, so very low window variance is
    # a useful proxy for disconnected or otherwise uninformative signals.
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

def build_dataset():
    """Build the configured Ruhrland multitask dataset template."""
    dataset = Ruhrlandklinik(
        channels=channels_cfgs,
        sample_frequency=sample_frequency,
        event_mapping=event_mapping,
        prepare_patient=prepare_sleep_staging_patient,
        prepare_target=partial(MultiLabelTrainer.prepare_target, task_config=normalized_task_config),
        prepare_sample=prepare_multilabel_sample,
        total_input=total_input,
        target_resolution=target_resolution,
    )
    return dataset


def summarize_patient_quality(patient, data_df, label_df, label_extra_df):
    """Summarize per-patient signal and label quality metrics for filtering."""
    if data_df is None or len(data_df) == 0 or label_df is None or len(label_df) == 0:
        return None

    durations = (label_df["Endtime"] - label_df["Starttime"]).dt.total_seconds()
    recording_hours = float((data_df.index[-1] - data_df.index[0]).total_seconds() / 3600.0)
    recording_hours = max(recording_hours, 1e-6)

    eeg_cols = [c for c in sleep_channels if c in data_df.columns]
    eeg_std = float(np.nanmedian([float(data_df[c].std()) for c in eeg_cols])) if len(eeg_cols) > 0 else 0.0

    chest_std = float(data_df["Chest"].std()) if "Chest" in data_df.columns else 0.0
    abdomen_std = float(data_df["Abdomen"].std()) if "Abdomen" in data_df.columns else 0.0
    pulse_std = float(data_df["Pulse Waveform"].std()) if "Pulse Waveform" in data_df.columns else 0.0

    if "Saturation" in data_df.columns:
        spo2 = data_df["Saturation"].astype(float)
        spo2_mean = float(spo2.mean())
        spo2_low_fraction = float((spo2 < 50).mean())
    else:
        spo2_mean = 0.0
        spo2_low_fraction = 1.0

    sleep_mask = label_df["Label"].isin(["n1", "n2", "n3", "rem"])
    arousal_mask = label_df["Label"] == "arousal"
    desat_mask = label_df["Label"] == "desaturation"
    apnea_mask = label_df["Label"] == "apnea"
    hypopnea_mask = label_df["Label"] == "hypopnea"

    sleep_hours = float(durations.loc[sleep_mask].sum() / 3600.0)
    arousal_rate_per_hour = float(arousal_mask.sum() / recording_hours)
    desat_rate_per_hour = float(desat_mask.sum() / recording_hours)
    respiratory_event_fraction = float(durations.loc[apnea_mask | hypopnea_mask].sum() / (recording_hours * 3600.0))

    return {
        "patient": patient,
        "recording_hours": recording_hours,
        "sleep_hours": sleep_hours,
        "eeg_std": eeg_std,
        "chest_std": chest_std,
        "abdomen_std": abdomen_std,
        "pulse_std": pulse_std,
        "spo2_mean": spo2_mean,
        "spo2_low_fraction": spo2_low_fraction,
        "arousal_rate_per_hour": arousal_rate_per_hour,
        "desat_rate_per_hour": desat_rate_per_hour,
        "respiratory_event_fraction": respiratory_event_fraction,
    }


def apply_patient_filters(dataset, patients):
    """Apply script-specific signal-QC and outlier filters to Ruhrland patients."""
    # TODO FROM HERE -> A bit too strong these filterings
    # TODO ONLY APPLY FOR TRAIN DATA?
    stats_df = dataset.get_patient_stats(patients, summarize_patient_quality, num_workers=num_workers_dataset)

    # First apply hard physiological / signal-quality checks. These remove
    # obviously broken recordings before we compute cohort-relative quantiles.
    filtered = stats_df[
        (stats_df["eeg_std"] > 1e-5)
        & (stats_df["chest_std"] > 1e-5)
        & (stats_df["abdomen_std"] > 1e-5)
        # & (stats_df["pulse_std"] > 500)
        # & (stats_df["spo2_mean"] >= 50.0)
        # & (stats_df["spo2_mean"] <= 100.0)
        & (stats_df["spo2_low_fraction"] <= 0.25)
        & (stats_df["sleep_hours"] > 0.5)
    ].copy()

    # Then remove distribution outliers. We do this after the hard QC pass so
    # the quantiles are estimated on already plausible PSGs rather than being
    # dominated by a few pathological recordings.
    for column in ["sleep_hours", "arousal_rate_per_hour", "desat_rate_per_hour", "respiratory_event_fraction"]:
        lower = filtered[column].quantile(patient_filter_quantile)
        upper = filtered[column].quantile(1 - patient_filter_quantile)
        filtered = filtered[(filtered[column] >= lower) & (filtered[column] <= upper)]

    kept = filtered["patient"].tolist()
    logger.info(
        f"Filtered Ruhrland multilabel patients: kept {len(kept)}/{len(patients)} after signal-QC and "
        f"task-distribution filtering."
    )
    return kept

def initialize_dataset(dataset, patients):
    """Filter patients and initialize the multitask dataset."""
    # patients = patients[:10]
    filtered_patients = apply_patient_filters(dataset, patients)
    dataset.initialize(filtered_patients, num_workers=num_workers_dataset)
    return dataset

def list_split_patients(purpose: str, dry_run: bool) -> list[str]:
    """List train or test patients with the required channel set."""
    edf_root = edf_folder if purpose == "train" else edf_folder_test
    patients = [p for p in get_edf_files_in_repo(edf_root, recursive=True) if has_required_channels(p)]
    return patients[:2] if dry_run else patients


def load_split_dataset(purpose: str, dry_run: bool):
    """Build and initialize one Ruhrland dataset split."""
    patients = list_split_patients(purpose, dry_run)
    return initialize_dataset(build_dataset(), patients)


def build_model(dataset):
    """Build the current multitask composite model for Ruhrland data."""
    respiratory_model = UTime(
        ts_len=dataset.get_timeseries_len(),
        n_channels=4,
        classes=None,
        sampling_frequency="0.01s", # TODO CHANGE
        channel=[16, 32, 64, 128],
        maxpool=[10, 8, 6, 4],
        kernel=[5, 5, 5, 5],
        norm="channel",
        mlp_size=64,
    )
    sleep_model = SleepTransformer(
        classes=None,
        n_channels=1,
    )

    input_channels = dataset.get_input_channels()
    missing = sorted(
        {
            channel
            for entry in [
                MetaModelEntry(respiratory_model, ["Chest", "Abdomen", "Saturation", "Pulse Waveform"]),
                MetaModelEntry(sleep_model, ["EEG"]),
            ]
            for channel in entry.input_channels
            if channel not in input_channels
        }
    )
    if len(missing) > 0:
        raise ValueError(f"Dataset is missing required MetaModel input channels: {missing}. Available: {input_channels}")

    model = MetaModel(
        task_config=normalized_task_config,
        input_channels=input_channels,
        models=[
            MetaModelEntry(respiratory_model, ["Chest", "Abdomen", "Saturation", "Pulse Waveform"]),
            MetaModelEntry(sleep_model, ["EEG"]),
        ],
    )
    return model

def main():
    """Build datasets, trainer, and model, then launch the multitask run."""
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    logger.add_sink(MlflowSink(tracking_uri="sqlite:///mlflow.sqlite", experiment=experiment_name))

    logger.context("Train")
    train_dataset = load_split_dataset("train", dry_run=False)
    logger.uncontext()

    logger.context("Test")
    test_dataset = load_split_dataset("test", dry_run=False)
    logger.uncontext()

    missing = sorted(set(train_dataset.get_classes()) - set(label for cfg in normalized_task_config.values() for label in cfg["labels"]))
    if len(missing) > 0:
        raise ValueError(f"Task config is missing dataset classes: {missing}")
    trainer = MultiLabelTrainer(
        epochs=epochs,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ),
        task_config=normalized_task_config,
        condition_task="sleep",
        condition_labels=["n1", "n2", "n3", "rem"],
        conditioned_tasks=["breathing", "arousal", "desat"],
        save_every=10,
        log_batches=False,
    )

    model = build_model(train_dataset)
    run(
        RunCfg(
            experiment_name=experiment_name,
            model_name="MetaModel",
            model=model,
            trainer=trainer,
            train_datasets=[train_dataset],
            val_datasets=[],
            test_datasets=[("test", test_dataset)],
            batch_size=batch_size,
            n_samples=n_samples,
            num_workers_dataloader=num_workers_dataloader,
            use_energy_tracker=False,
            collate_fn=batch_collate,
        )
    )


if __name__ == "__main__":
    main()
