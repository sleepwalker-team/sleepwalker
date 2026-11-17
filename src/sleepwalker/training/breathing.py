"""Task-specific preparation callbacks for breathing configurations."""

from __future__ import annotations

import pandas as pd

from sleepwalker.trainer.utils.filtering import trim_event


def _clip_to_intervals(
    df: pd.DataFrame | None,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame | None:
    if df is None or len(df) == 0 or len(intervals) == 0:
        return None
    clipped_rows = []
    for row in df.itertuples(index=False):
        row_start = pd.Timestamp(row.Starttime)
        row_end = pd.Timestamp(row.Endtime)
        for keep_start, keep_end in intervals:
            clipped_start = max(row_start, keep_start)
            clipped_end = min(row_end, keep_end)
            if clipped_start >= clipped_end:
                continue
            clipped_row = dict(zip(df.columns, row))
            clipped_row["Starttime"] = clipped_start
            clipped_row["Endtime"] = clipped_end
            clipped_rows.append(clipped_row)
    if not clipped_rows:
        return None
    return pd.DataFrame(clipped_rows, columns=df.columns)


def prepare_patient(
    label_df,
    label_extra_df,
    patient=None,
    *,
    min_desaturation_sleep_overlap_seconds: float = 1.0,
):
    """Retain sleep intervals overlapping a desaturation for breathing labels."""
    if label_df is None:
        return None, label_extra_df
    trimmed = trim_event(label_df, label_extra_df, ["sleep"])
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed

    trimmed = trim_event(label_df, label_extra_df, ["desaturation"])
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed

    labels = label_df.copy()
    labels["Starttime"] = pd.to_datetime(labels["Starttime"])
    labels["Endtime"] = pd.to_datetime(labels["Endtime"])
    desaturations = labels[labels["Label"].eq("desaturation")].sort_values(
        ["Starttime", "Endtime"]
    )
    sleep = labels[labels["Label"].eq("sleep")].sort_values(["Starttime", "Endtime"])
    if len(desaturations) == 0 or len(sleep) == 0:
        return None

    overlap_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    sleep_rows = list(sleep.itertuples(index=False))
    sleep_idx = 0
    for desaturation in desaturations.itertuples(index=False):
        desat_start = pd.Timestamp(desaturation.Starttime)
        desat_end = pd.Timestamp(desaturation.Endtime)
        while sleep_idx < len(sleep_rows) and pd.Timestamp(sleep_rows[sleep_idx].Endtime) <= desat_start:
            sleep_idx += 1
        current = sleep_idx
        while current < len(sleep_rows):
            sleep_start = pd.Timestamp(sleep_rows[current].Starttime)
            sleep_end = pd.Timestamp(sleep_rows[current].Endtime)
            if sleep_start >= desat_end:
                break
            overlap_start = max(desat_start, sleep_start)
            overlap_end = min(desat_end, sleep_end)
            if (
                overlap_end > overlap_start
                and (overlap_end - overlap_start).total_seconds()
                >= float(min_desaturation_sleep_overlap_seconds)
            ):
                overlap_intervals.append((overlap_start, overlap_end))
            current += 1

    clipped = _clip_to_intervals(labels, overlap_intervals)
    if clipped is None or len(clipped) == 0:
        return None
    return clipped, label_extra_df
