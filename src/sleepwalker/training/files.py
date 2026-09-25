"""EDF selectors shared by Python and YAML standard training workflows."""

from __future__ import annotations

from collections import Counter
import multiprocessing
from pathlib import Path
from typing import Sequence

import pandas as pd

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets.Basedataset import BaseDataset
from sleepwalker.utils import logger


PAP_CHANNEL_PATTERNS = (
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
)

def edf_filter_result(
    item: tuple[
        str,
        tuple[tuple[str, ...], ...],
        float | None,
        tuple[str, ...],
    ],
) -> tuple[str, str]:
    """Check one EDF header and return its path plus a rejection reason."""
    edf_path, required_channel_groups, min_duration_s, excluded_channels = item
    if required_channel_groups or min_duration_s is not None or excluded_channels:
        try:
            metadata = read_edf_meta(edf_path)
        except Exception:
            return edf_path, "metadata_error"

        if min_duration_s is not None and float(metadata["duration_s"]) < min_duration_s:
            return edf_path, "too_short"
        available_signals = set(metadata["signals"])

        if any( not any(channel in available_signals for channel in alternatives) for alternatives in required_channel_groups ):
            return edf_path, "missing_required_channel"
        if any(channel in available_signals for channel in excluded_channels):
            return edf_path, "excluded_channel"

    return edf_path, "usable"


def filter_edf_files(
    patients: Sequence[str | Path],
    *,
    dataset: BaseDataset,
    require_channels: bool = True,
    min_duration: str | None = None,
    excluded_channels: Sequence[str] | None = None,
    num_workers: int = 1,
) -> list[str]:
    """Filter EDF paths using only signal-header metadata.

    By default, every configured logical dataset channel must have at least one
    physical alternative in the EDF. ``min_duration`` accepts pandas duration
    strings such as ``"30min"``. ``excluded_channels`` rejects files containing
    any listed physical signal. Annotation contents are deliberately outside
    this generic filter.
    """

    min_duration_s = None if min_duration is None else pd.Timedelta(min_duration).total_seconds()

    patient_list = [str(path) for path in patients]
    required_channel_groups = ( tuple(tuple(channel.physical_names) for channel in dataset.channels) if require_channels else () )
    work = [(edf_path, required_channel_groups, None if min_duration_s is None else float(min_duration_s), tuple(excluded_channels or ())) for edf_path in patient_list]

    worker_count = int(num_workers)
    if worker_count > 1 and work:
        with multiprocessing.Pool(worker_count) as pool:
            results = list(pool.imap(edf_filter_result, work))
    else:
        results = [edf_filter_result(item) for item in work]

    reasons = Counter(reason for _, reason in results)
    selected = [edf_path for edf_path, reason in results if reason == "usable"]
    excluded = { reason: count for reason, count in sorted(reasons.items()) if reason != "usable" }
    logger.info(
        f"Patient filter kept {len(selected)}/{len(patient_list)} EDF files; "
        f"excluded by reason: {excluded or '{}'}"
    )
    return selected
