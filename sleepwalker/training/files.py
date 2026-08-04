"""EDF selectors shared by Python and YAML standard training workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets.Basedataset import BaseDataset
from sleepwalker.datasets.HSP import (
    get_annotated_hsp_edf_files,
    get_hsp_annotation_label_counts,
    get_hsp_annotation_path,
)
from sleepwalker.datasets.utils import get_edf_files_in_repo
from sleepwalker.trainer.utils.filtering import filter_patients_by_sleep_time


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

SLEEP_STAGING_LABELS = ("n1", "n2", "n3", "rem")


def has_required_channels(
    edf_path: str | Path,
    dataset: BaseDataset,
    *,
    available_signals: Sequence[str] | None = None,
) -> bool:
    """Return whether every logical dataset channel has a physical EDF alternative."""
    available = set(
        available_signals
        if available_signals is not None
        else read_edf_meta(str(edf_path))["signals"]
    )
    return all(
        any(name in available for name in channel.physical_names)
        for channel in dataset.channels
    )


def is_pap_file(
    edf_path: str | Path,
    *,
    pap_channel_patterns: Sequence[str] = PAP_CHANNEL_PATTERNS,
    available_signals: Sequence[str] | None = None,
) -> bool:
    """Detect a PAP recording from the known Ruhrland PAP signal inventory."""
    available = set(
        available_signals
        if available_signals is not None
        else read_edf_meta(str(edf_path))["signals"]
    )
    return any(pattern in available for pattern in pap_channel_patterns)


def select_edf_files(
    root: str | Path,
    *,
    dataset: BaseDataset,
    recursive: bool = True,
    require_channels: bool = True,
    include_pap: bool = True,
    annotated_only: bool = False,
    required_annotation_labels: Sequence[str] | None = None,
    max_files: int | None = None,
) -> list[str]:
    """Select EDFs using common channel, PAP, and HSP-annotation criteria."""
    if annotated_only:
        files = get_annotated_hsp_edf_files(root, recursive=recursive)
    else:
        files = get_edf_files_in_repo(root, recursive=recursive)

    selected: list[str] = []
    for edf_path in files:
        available_signals = None
        if require_channels or not include_pap:
            available_signals = read_edf_meta(str(edf_path))["signals"]
        if require_channels and not has_required_channels(
            edf_path,
            dataset,
            available_signals=available_signals,
        ):
            continue
        if not include_pap and is_pap_file(
            edf_path,
            available_signals=available_signals,
        ):
            continue
        if required_annotation_labels:
            annotation_path = get_hsp_annotation_path(edf_path)
            if annotation_path is None:
                continue
            counts = get_hsp_annotation_label_counts(annotation_path)
            if not any(int(counts.get(label, 0)) > 0 for label in required_annotation_labels):
                continue
        selected.append(str(edf_path))
        if max_files is not None and len(selected) >= int(max_files):
            break
    return selected


def select_sleep_staging_files(
    root: str | Path,
    *,
    dataset: BaseDataset,
    training: bool,
    recursive: bool = True,
    sleep_labels: Sequence[str] = SLEEP_STAGING_LABELS,
    sleep_time_filter_quantile: float = 0.05,
    num_workers: int = 8,
    max_files: int | None = None,
) -> list[str]:
    """Select channel-compatible EDFs and filter training sleep-time outliers."""
    selected = select_edf_files(
        root,
        dataset=dataset,
        recursive=recursive,
        require_channels=True,
        max_files=max_files,
    )
    if training and float(sleep_time_filter_quantile) > 0:
        selected = filter_patients_by_sleep_time(
            selected,
            dataset=dataset,
            sleep_labels=sleep_labels,
            quantile=float(sleep_time_filter_quantile),
            num_workers=int(num_workers),
            label=dataset.__class__.__name__,
        )
    return selected

