"""Small event-window filtering helpers shared by training scripts."""

from __future__ import annotations

from functools import partial
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _summarize_sleep_time(
    patient: str,
    label_df: Optional[pd.DataFrame],
    *,
    sleep_labels: tuple[str, ...],
    **_kwargs,
) -> Optional[dict[str, float | str]]:
    if label_df is None or len(label_df) == 0:
        return None
    required = {"Starttime", "Endtime", "Label"}
    missing = required.difference(label_df.columns)
    if missing:
        raise ValueError(f"Cannot summarize sleep time; missing label columns {sorted(missing)}.")
    duration_s = (label_df["Endtime"] - label_df["Starttime"]).dt.total_seconds()
    sleep_seconds = duration_s[label_df["Label"].isin(sleep_labels)].sum()
    return {"patient": patient, "sleep_seconds": float(sleep_seconds)}


def filter_patients_by_sleep_time(
    patients: Sequence[str],
    dataset,
    sleep_labels: Sequence[str],
    quantile: float,
    num_workers: int,
    label: str,
) -> list[str]:
    """Drop symmetric sleep-duration outliers using patient-level statistics."""
    patients = list(patients)
    if not 0.0 <= quantile < 0.5:
        raise ValueError("quantile must be in [0, 0.5).")
    if len(patients) < 3:
        return patients
    stats = dataset.get_patient_stats(
        patients,
        partial(_summarize_sleep_time, sleep_labels=tuple(sleep_labels)),
        num_workers=num_workers,
    )
    if len(stats) < 3 or "sleep_seconds" not in stats.columns:
        return patients
    lower = stats["sleep_seconds"].quantile(quantile)
    upper = stats["sleep_seconds"].quantile(1.0 - quantile)
    return stats.loc[
        stats["sleep_seconds"].between(lower, upper, inclusive="both"),
        "patient",
    ].tolist()


def trim_event(
    data_df: Optional[pd.DataFrame],
    label_df: Optional[pd.DataFrame],
    label_extra_df: Optional[pd.DataFrame],
    keep_events: Sequence[str] | None = None,
) -> Optional[tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
    """Trim leading/trailing rows until retained events occur.

    Keeps the interval from the first row whose label is in `keep_events`
    to the last row whose label is in `keep_events`.

    Args:
        data_df: Currently unused; kept for compatibility.
        label_df: Primary event table.
        label_extra_df: Optional secondary event table clipped to the retained interval.
        keep_events: Labels that define the retained region. When omitted, the
            first and last non-wake rows are used.

    Returns:
        `(trimmed_label_df, trimmed_label_extra_df)` or `None` if no keep-event occurs.
    """
    if label_df is None:
        return None, label_extra_df
    if len(label_df) == 0:
        return None

    label_df = label_df.sort_values(["Starttime", "Endtime"]).reset_index(drop=True)
    if keep_events is None:
        keep_mask = ~label_df["Label"].isin({"wake", "wach"})
    else:
        keep_events = set(keep_events)
        if not keep_events:
            return None
        keep_mask = label_df["Label"].isin(keep_events)
    if not keep_mask.any():
        return None

    keep_positions = np.flatnonzero(keep_mask.to_numpy())
    first_idx = int(keep_positions[0])
    last_idx = int(keep_positions[-1])

    label_df = label_df.iloc[first_idx:last_idx + 1].copy()

    lower = label_df["Starttime"].min()
    upper = label_df["Endtime"].max()

    if label_extra_df is not None:
        label_extra_df = label_extra_df.copy()
        label_extra_df = label_extra_df[
            (label_extra_df["Endtime"] > lower)
            & (label_extra_df["Starttime"] < upper)
        ]

        if len(label_extra_df) > 0:
            label_extra_df["Starttime"] = label_extra_df["Starttime"].clip(
                lower=lower, upper=upper
            )
            label_extra_df["Endtime"] = label_extra_df["Endtime"].clip(
                lower=lower, upper=upper
            )
            label_extra_df = label_extra_df[
                label_extra_df["Endtime"] > label_extra_df["Starttime"]
            ]
        else:
            label_extra_df = None

    return label_df, label_extra_df
