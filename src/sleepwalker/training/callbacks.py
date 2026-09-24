"""Common dataset callbacks used by standard training configurations."""

from __future__ import annotations

from typing import Sequence

from sleepwalker.datasets.Basedataset import prepare_tensor_sample
from sleepwalker.models.preprocessors.RobustScaler import RobustScaler
from sleepwalker.trainer.utils.filtering import trim_event


# A small number of existing expert packages reference the former symbol.
tensor_sample = prepare_tensor_sample


def trim_patient_to_events(
    label_df,
    label_extra_df,
    patient=None,
    *,
    keep_events: Sequence[str] | None = None,
):
    """Adapt :func:`trim_event` to the ``prepare_patient`` callback contract."""
    return trim_event(label_df, label_extra_df, keep_events)


def prepare_patient_events(
    label_df,
    label_extra_df,
    patient=None,
    *,
    min_seconds: float,
    labels: Sequence[str],
    keep_events: Sequence[str] | None = None,
):
    """Remove short selected events and optionally trim to retained events."""
    minimum_duration = float(min_seconds)
    if minimum_duration < 0:
        raise ValueError("min_seconds must not be negative.")
    duration_filtered_labels = set(labels)

    def filter_events(event_df):
        if event_df is None or not duration_filtered_labels:
            return event_df
        required_columns = {"Label", "Starttime", "Endtime"}
        missing_columns = required_columns.difference(event_df.columns)
        if missing_columns:
            raise ValueError(f"Cannot filter event durations; missing columns {sorted(missing_columns)}.")
        durations = (event_df["Endtime"] - event_df["Starttime"]).dt.total_seconds()
        short_event_mask = event_df["Label"].isin(duration_filtered_labels) & ~durations.ge(minimum_duration)
        return event_df.loc[~short_event_mask].copy()

    label_df = filter_events(label_df)
    label_extra_df = filter_events(label_extra_df)
    if keep_events is not None:
        return trim_event(label_df, label_extra_df, keep_events)
    return label_df, label_extra_df


def robust_scaler(
    *,
    n_channels: int,
    skip_first: int = 0,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
) -> RobustScaler:
    """Construct a scaler for a dataset-derived channel range."""
    return RobustScaler(
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        channels=list(range(int(skip_first), int(n_channels))),
    )
