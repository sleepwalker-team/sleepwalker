from __future__ import annotations

from functools import partial
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from sleepwalker.utils import logger


def trim_wake(
    data_df: Optional[pd.DataFrame],
    label_df: Optional[pd.DataFrame],
    label_extra_df: Optional[pd.DataFrame],
    wake_label: str = "wake",
) -> Optional[tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    if label_df is None or len(label_df) == 0:
        return None

    label_df = label_df.sort_values(["Starttime", "Endtime"]).reset_index(drop=True)
    non_wake = label_df["Label"] != wake_label
    if not non_wake.any():
        return None

    non_wake_positions = np.flatnonzero(non_wake.to_numpy())
    first_idx = int(non_wake_positions[0])
    last_idx = int(non_wake_positions[-1])
    label_df = label_df.iloc[first_idx:last_idx + 1].copy()

    lower = label_df["Starttime"].min()
    upper = label_df["Endtime"].max()

    if label_extra_df is not None:
        label_extra_df = label_extra_df.copy()
        label_extra_df = label_extra_df[
            (label_extra_df["Endtime"] > lower) & (label_extra_df["Starttime"] < upper)
        ]
        if len(label_extra_df) > 0:
            label_extra_df["Starttime"] = label_extra_df["Starttime"].clip(lower=lower, upper=upper)
            label_extra_df["Endtime"] = label_extra_df["Endtime"].clip(lower=lower, upper=upper)
            label_extra_df = label_extra_df[label_extra_df["Endtime"] > label_extra_df["Starttime"]]
        else:
            label_extra_df = None

    return label_df, label_extra_df


def summarize_patient_sleep_time(
    patient: str,
    label_df: pd.DataFrame | None,
    sleep_labels: Sequence[str],
    **_kwargs,
) -> dict[str, float | str] | None:
    if label_df is None or len(label_df) == 0:
        return None

    trimmed_events = label_df.copy()
    trimmed_events["duration_s"] = (trimmed_events["Endtime"] - trimmed_events["Starttime"]).dt.total_seconds()
    sleep_seconds = trimmed_events.loc[trimmed_events["Label"].isin(list(sleep_labels)), "duration_s"].sum()
    return {"patient": patient, "sleep_seconds": float(sleep_seconds)}


def filter_patients_by_sleep_time(
    patients: list[str],
    dataset,
    sleep_labels: Sequence[str],
    quantile: float,
    num_workers: int,
    label: str,
) -> list[str]:
    if len(patients) < 3:
        return patients

    stats_df = dataset.get_patient_stats(
        patients,
        partial(summarize_patient_sleep_time, sleep_labels=sleep_labels),
        num_workers=num_workers,
    )
    if len(stats_df) < 3 or "sleep_seconds" not in stats_df.columns:
        return patients

    lower = stats_df["sleep_seconds"].quantile(quantile)
    upper = stats_df["sleep_seconds"].quantile(1 - quantile)
    filtered = stats_df[(stats_df["sleep_seconds"] >= lower) & (stats_df["sleep_seconds"] <= upper)]["patient"].tolist()
    logger.info(
        f"Filtered sleep-time outliers for {label}: kept {len(filtered)}/{len(patients)} patients "
        f"after dropping the bottom/top {quantile:.0%}."
    )
    return filtered
