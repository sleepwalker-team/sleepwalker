"""Common dataset callbacks used by standard training configurations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from sleepwalker.datasets.Basedataset import prepare_tensor_sample
from sleepwalker.datasets.events import exclude_events_on_wake, merge_event_intervals, validate_event_table
from sleepwalker.models.preprocessors.RobustScaler import RobustScaler
from sleepwalker.trainer.utils.filtering import trim_event


# A small number of existing expert packages reference the former symbol.
tensor_sample = prepare_tensor_sample


def map_events(label_df: pd.DataFrame, label_extra_df: pd.DataFrame | None, *, mapping: Mapping[str, str], remove_unmapped: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Map both annotators independently, retaining intervals and all metadata.

    Unmatched labels are removed only when requested. This helper does not
    construct negative intervals or derive annotation/model vocabularies.
    """
    if not isinstance(mapping, Mapping):
        raise TypeError("mapping must be a dictionary of event labels.")
    for label in [*mapping.keys(), *mapping.values()]:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("mapping keys and values must be non-empty strings.")
    if not isinstance(remove_unmapped, bool):
        raise TypeError("remove_unmapped must be a boolean.")

    def transform(events, annotator):
        if events is None:
            return None
        validate_event_table(events, context=annotator)
        result = events.loc[events["Label"].isin(mapping)].copy() if remove_unmapped else events.copy()
        result["Label"] = result["Label"].map(lambda label: mapping.get(label, label))
        return result

    if label_df is None:
        raise TypeError("primary annotations must be a pandas DataFrame.")
    return transform(label_df, "primary annotations"), transform(label_extra_df, "alternate annotations")


def prepare_sleep_events(label_df, label_extra_df, *, mapping, merge_labels=(), exclude_on_wake=(), touching_counts_as_overlap=False, remove_unmapped=True):
    """Merge, exclude on wake, then map each annotator's own physiological events.

    The same configured policy applies to both sources; evidence is never
    shared between annotators. The input tables are left unchanged.
    """
    def transform(events):
        if events is None:
            return None
        events = merge_event_intervals(events, labels=merge_labels, touching_counts_as_overlap=touching_counts_as_overlap)
        return exclude_events_on_wake(events, labels=exclude_on_wake, touching_counts_as_overlap=touching_counts_as_overlap)

    return map_events(transform(label_df), transform(label_extra_df), mapping=mapping, remove_unmapped=remove_unmapped)


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
