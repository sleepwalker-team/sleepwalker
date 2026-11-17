"""Ruhrland multilabel cohort selection for standard training configs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets.Basedataset import BaseDataset
from sleepwalker.datasets.utils import get_edf_files_in_repo
from sleepwalker.training.files import PAP_CHANNEL_PATTERNS
from sleepwalker.utils import logger


def has_all_configured_channels(
    edf_path: str | Path,
    dataset: BaseDataset,
    *,
    available_signals: list[str] | None = None,
) -> bool:
    """Require every configured signal and companion quality channel."""
    available = set(
        available_signals
        if available_signals is not None
        else read_edf_meta(str(edf_path))["signals"]
    )
    required = {
        signal
        for channel in dataset.channels
        for physical_name in channel.physical_names
        for signal in (physical_name, channel.quality_name_for(physical_name))
        if signal is not None
    }
    return required.issubset(available)


def summarize_patient_quality(patient, data_df, label_df, label_extra_df):
    """Summarize the signal and task-distribution metrics used for cohort QC."""
    if data_df is None or len(data_df) == 0 or label_df is None or len(label_df) == 0:
        return None

    durations = (label_df["Endtime"] - label_df["Starttime"]).dt.total_seconds()
    recording_hours = float((data_df.index[-1] - data_df.index[0]).total_seconds() / 3600.0)
    recording_hours = max(recording_hours, 1e-6)

    eeg_columns = [
        column
        for column in ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"]
        if column in data_df.columns
    ]
    eeg_std = (
        float(np.nanmedian([float(data_df[column].std()) for column in eeg_columns]))
        if eeg_columns
        else 0.0
    )
    chest_std = float(data_df["Chest"].std()) if "Chest" in data_df.columns else 0.0
    abdomen_std = float(data_df["Abdomen"].std()) if "Abdomen" in data_df.columns else 0.0

    if "Saturation" in data_df.columns:
        saturation = data_df["Saturation"].astype(float)
        saturation_low_fraction = float((saturation < 50).mean())
    else:
        saturation_low_fraction = 1.0

    labels = label_df["Label"]
    sleep_mask = labels.isin(["n1", "n2", "n3", "rem"])
    sleep_hours = float(durations.loc[sleep_mask].sum() / 3600.0)
    return {
        "patient": patient,
        "sleep_hours": sleep_hours,
        "eeg_std": eeg_std,
        "chest_std": chest_std,
        "abdomen_std": abdomen_std,
        "saturation_low_fraction": saturation_low_fraction,
        "arousal_rate_per_hour": float(labels.eq("arousal").sum() / recording_hours),
        "desaturation_rate_per_hour": float(
            labels.eq("desaturation").sum() / recording_hours
        ),
        "respiratory_event_fraction": float(
            durations.loc[labels.isin(["apnea", "hypopnea"])].sum()
            / (recording_hours * 3600.0)
        ),
    }


def filter_patient_cohort(
    patients: list[str],
    *,
    dataset: BaseDataset,
    quantile: float = 0.05,
    num_workers: int = 8,
) -> list[str]:
    """Apply the existing Ruhrland signal-QC and cohort-outlier filters."""
    if not 0.0 <= float(quantile) < 0.5:
        raise ValueError("quantile must be in [0, 0.5).")
    if not patients:
        return []

    stats = dataset.get_patient_stats(
        patients,
        summarize_patient_quality,
        num_workers=int(num_workers),
    )
    if len(stats) == 0:
        return []
    filtered = stats[
        (stats["eeg_std"] > 1e-5)
        & (stats["chest_std"] > 1e-5)
        & (stats["abdomen_std"] > 1e-5)
        & (stats["saturation_low_fraction"] <= 0.25)
        & (stats["sleep_hours"] > 0.5)
    ].copy()

    for column in [
        "sleep_hours",
        "arousal_rate_per_hour",
        "desaturation_rate_per_hour",
        "respiratory_event_fraction",
    ]:
        if len(filtered) == 0:
            break
        lower = filtered[column].quantile(float(quantile))
        upper = filtered[column].quantile(1.0 - float(quantile))
        filtered = filtered[filtered[column].between(lower, upper, inclusive="both")]

    selected = filtered["patient"].tolist()
    logger.info(
        f"Filtered Ruhrland multilabel patients: kept {len(selected)}/{len(patients)} "
        "after signal-QC and task-distribution filtering."
    )
    return selected


def select_files(
    root: str | Path,
    *,
    dataset: BaseDataset,
    training: bool,
    recursive: bool = True,
    include_pap: bool = True,
    quantile: float = 0.05,
    num_workers: int = 8,
    max_candidates: int | None = None,
) -> list[str]:
    """Discover compatible Ruhrland files and optionally apply training QC."""
    candidates = get_edf_files_in_repo(root, recursive=recursive)
    if max_candidates is not None:
        candidates = candidates[: int(max_candidates)]

    selected = []
    for edf_path in candidates:
        signals = read_edf_meta(str(edf_path))["signals"]
        if not has_all_configured_channels(
            edf_path,
            dataset,
            available_signals=signals,
        ):
            continue
        if not include_pap and any(pattern in signals for pattern in PAP_CHANNEL_PATTERNS):
            continue
        selected.append(str(edf_path))

    if training:
        selected = filter_patient_cohort(
            selected,
            dataset=dataset,
            quantile=float(quantile),
            num_workers=int(num_workers),
        )
    return selected
