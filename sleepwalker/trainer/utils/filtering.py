"""Small event-window filtering helpers shared by training scripts."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def trim_event(
    data_df: Optional[pd.DataFrame],
    label_df: Optional[pd.DataFrame],
    label_extra_df: Optional[pd.DataFrame],
    keep_events: Sequence[str] | None = None,
) -> Optional[tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
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
    if label_df is None or len(label_df) == 0:
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
