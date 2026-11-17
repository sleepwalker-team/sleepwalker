"""Shared annotation-table contracts and source-independent interval operations.

Event names are extensible strings, rather than a closed enum. The catalogue
in docs/reference/events.md defines the recognized physiological vocabulary.
These functions never interpret SourceLabel or infer missing measurements.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


EVENT_COLUMNS = ["Label", "Starttime", "Endtime"]
SLEEP_STAGE_LABELS = ["wake", "n1", "n2", "n3", "rem", "unscored"]


def decode_event_labels(events: pd.DataFrame, *, mapping: Mapping[str, str]) -> pd.DataFrame:
    """Decode explicit reader source terms, preserving unclassified text.

    Readers supply their own source dictionary. Unknown meanings become
    other; the original source text remains available for investigation.
    This is source decoding, not the task-level prepare_events callback.
    """
    result = events.copy()
    if result.empty:
        for column in ("Starttime", "Endtime"):
            result[column] = pd.to_datetime(result[column])
    if "SourceLabel" not in result:
        result["SourceLabel"] = result["Label"].astype("string")
    source = result["Label"].astype("string").str.strip().str.lower()
    result["Label"] = source.map(mapping).fillna("other")
    return result


def validate_event_labels(labels: Sequence[str], *, context: str = "event labels", allow_empty: bool = False) -> list[str]:
    """Validate an ordered vocabulary without sorting or discovering labels."""
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise TypeError(f"{context} must be an ordered sequence of strings.")
    result = list(labels)
    if not allow_empty and not result:
        raise ValueError(f"{context} must not be empty.")
    if any(not isinstance(label, str) or not label.strip() for label in result):
        raise ValueError(f"{context} must contain non-empty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"{context} must contain unique labels.")
    return result


def validate_event_table(events: pd.DataFrame, *, allowed_labels: Sequence[str] | None = None, context: str = "events") -> None:
    """Check the temporal core; optional missing metadata never removes rows.

    Empty tables still require typed datetime columns. Readers must decode
    source clocks before this boundary; strings and numeric offsets are errors.
    """
    if not isinstance(events, pd.DataFrame):
        raise TypeError(f"{context} must be a pandas DataFrame.")
    if not events.columns.is_unique:
        raise ValueError(f"{context} contains duplicate column names.")
    missing = sorted(set(EVENT_COLUMNS) - set(events.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns {missing}.")
    if any(not isinstance(label, str) or not label.strip() for label in events["Label"]):
        raise ValueError(f"{context} Label values must be non-empty strings.")
    for column in ("Starttime", "Endtime"):
        if not pd.api.types.is_datetime64_any_dtype(events[column].dtype) or events[column].isna().any():
            raise ValueError(f"{context} {column} must contain non-missing pandas timestamps.")
    if events["Starttime"].dt.tz != events["Endtime"].dt.tz:
        raise ValueError(f"{context} Starttime and Endtime must use the same timezone.")
    if not (events["Endtime"] > events["Starttime"]).all():
        raise ValueError(f"{context} requires Endtime later than Starttime (half-open intervals).")
    if allowed_labels is not None:
        vocabulary = validate_event_labels(allowed_labels, context=f"{context} allowed labels", allow_empty=True)
        unknown = sorted(set(events["Label"]) - set(vocabulary))
        if unknown:
            raise ValueError(f"{context} contains undeclared labels {unknown}.")


def merge_event_intervals(events: pd.DataFrame, *, labels: Sequence[str], touching_counts_as_overlap: bool = False) -> pd.DataFrame:
    """Union overlapping intervals independently for each selected label.

    Metadata is retained only if every contributing row agrees, including
    missingness; conflicting values become missing. In particular, a merged
    interval never arbitrarily inherits one event's desaturation measurement.
    Unselected events are retained individually. Output is chronological.
    """
    validate_event_table(events)
    if not isinstance(touching_counts_as_overlap, bool):
        raise TypeError("touching_counts_as_overlap must be a boolean.")
    selected = validate_event_labels(labels, context="merge labels", allow_empty=True)
    metadata = [column for column in events.columns if column not in [*EVENT_COLUMNS, "Duration"]]
    parts = [events.loc[~events["Label"].isin(selected)].copy()]
    for label in selected:
        group = events.loc[events["Label"] == label].sort_values(["Starttime", "Endtime"], kind="stable")
        rows = []
        current = None
        for row in group.to_dict("records"):
            overlaps = current is not None and (row["Starttime"] <= current["Endtime"] if touching_counts_as_overlap else row["Starttime"] < current["Endtime"])
            if overlaps:
                current["Endtime"] = max(current["Endtime"], row["Endtime"])
                for column in metadata:
                    left, right = current[column], row[column]
                    both_missing = pd.isna(left) and pd.isna(right)
                    both_equal = not pd.isna(left) and not pd.isna(right) and left == right
                    if not (both_missing or both_equal):
                        current[column] = pd.NA
            else:
                if current is not None:
                    rows.append(current)
                current = row
        if current is not None:
            rows.append(current)
        if rows:
            parts.append(pd.DataFrame(rows, columns=events.columns))
    result = pd.concat(parts, ignore_index=True).sort_values(["Starttime", "Endtime"], kind="stable").reset_index(drop=True)
    if "Duration" in result:
        durations = result["Endtime"] - result["Starttime"]
        result["Duration"] = durations if pd.api.types.is_timedelta64_dtype(events["Duration"]) else durations.dt.total_seconds()
    return result


def exclude_events_on_wake(events: pd.DataFrame, *, labels: Sequence[str], touching_counts_as_overlap: bool = False) -> pd.DataFrame:
    """Remove entire selected events overlapping wake; empty wake removes none.

    Default overlap follows half-open intervals. Set the explicit touching flag
    to preserve Ruhrland's historical inclusive endpoint exclusion policy.
    """
    validate_event_table(events)
    if not isinstance(touching_counts_as_overlap, bool):
        raise TypeError("touching_counts_as_overlap must be a boolean.")
    selected = validate_event_labels(labels, context="wake exclusion labels", allow_empty=True)
    wake = events.loc[events["Label"] == "wake"].sort_values("Starttime")
    if wake.empty or not selected:
        return events.copy()
    starts = wake["Starttime"].dt.as_unit("ns").astype("int64").to_numpy()
    ends = np.maximum.accumulate(wake["Endtime"].dt.as_unit("ns").astype("int64").to_numpy())
    candidates = events["Label"].isin(selected).to_numpy()
    event_starts = events["Starttime"].dt.as_unit("ns").astype("int64").to_numpy()
    event_ends = events["Endtime"].dt.as_unit("ns").astype("int64").to_numpy()
    previous = np.searchsorted(starts, event_ends, side="right" if touching_counts_as_overlap else "left") - 1
    has_previous = previous >= 0
    last_end = ends[np.maximum(previous, 0)]
    overlaps = last_end >= event_starts if touching_counts_as_overlap else last_end > event_starts
    return events.loc[~(candidates & has_previous & overlaps)].copy()
