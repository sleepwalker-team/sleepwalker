"""Model results per session and their conversion to API-facing records.

A result's frame holds `prob__<class>` columns plus `time` (epoch and event:
interval center; continuous: sample time). Events are derived from the intervals.
"""

import math
import re
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from threading import RLock

import numpy as np
import pandas as pd


class PredictionKind(str, Enum):
    EPOCH = "epoch"
    CONTINUOUS = "continuous"
    EVENT = "event"


@dataclass(frozen=True)
class PredictionResult:
    session_id: str
    model_id: str
    task: str
    classes: tuple[str, ...]
    resolution: float
    frame: pd.DataFrame
    kind: PredictionKind = PredictionKind.EPOCH


@dataclass(frozen=True)
class IntervalPrediction:
    """One fixed interval of an epoch or event task."""
    start: float
    end: float
    label: str
    probabilities: dict[str, float]


@dataclass(frozen=True)
class PointPrediction:
    """One sample of a continuous prediction."""
    time: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class ModelEvent:
    """Consecutive intervals whose most likely class is the same non-background class."""
    id: str
    start: float
    end: float
    label: str
    confidence: float

def background_class(classes: Sequence[str]) -> str | None:
    """The single "nothing happens" class of an event task."""
    background = [label for label in classes if re.match(r"(no(?:[ _-]|$)|kein|regular breathing$|none$)", label, re.IGNORECASE)]
    return background[0] if len(background) == 1 else None


class PredictionStore:
    """Holds at most one result per session and task."""

    def __init__(self) -> None:
        self._results: dict[tuple[str, str], PredictionResult] = {}
        self._mapped: dict[tuple[str, str], tuple] = {}
        self._lock = RLock()

    def add(self, result: PredictionResult) -> None:
        self.add_many([result])

    def add_many(self, results: list[PredictionResult]) -> None:
        """Publish all results or, on any conflict, none of them."""
        with self._lock:
            seen_models = {(result.session_id, result.model_id) for result in self._results.values()}
            seen_tasks = {(result.session_id, result.task) for result in self._results.values()}
            for result in results:
                model_key = (result.session_id, result.model_id)
                task_key = (result.session_id, result.task)
                if model_key in seen_models:
                    raise ValueError(f"Predictions already exist for session {result.session_id} and model {result.model_id}.")
                if task_key in seen_tasks:
                    raise ValueError(f"Predictions already exist for session {result.session_id} and task {result.task}.")
                seen_models.add(model_key)
                seen_tasks.add(task_key)

            for result in results:
                self._results[(result.session_id, result.task)] = replace(result, frame=result.frame.copy(deep=True))

    def get_by_task(self, session_id: str, task: str) -> PredictionResult | None:
        with self._lock:
            return self._results.get((session_id, task))

    def mapped(self, result: PredictionResult, recording_start: datetime) -> tuple[list[IntervalPrediction | PointPrediction], list[ModelEvent]]:
        """Records and events of a result, computed once: results are replaced, never changed in place."""
        key = (result.session_id, result.task)
        with self._lock:
            cached = self._mapped.get(key)
        if cached is None or cached[0] is not result:
            cached = (result, map_predictions(result, recording_start), model_events(result, recording_start))
            with self._lock:
                self._mapped[key] = cached
        return cached[1], cached[2]

    def list_for_session(self, session_id: str) -> list[PredictionResult]:
        with self._lock:
            results = [result for (stored_session_id, _), result in self._results.items() if stored_session_id == session_id]
        return sorted(results, key=lambda result: result.model_id)

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            keys = [key for key in self._results if key[0] == session_id]
            for key in keys:
                del self._results[key]
                self._mapped.pop(key, None)
            return len(keys)


def map_predictions(result: PredictionResult, recording_start: datetime) -> list[IntervalPrediction | PointPrediction]:
    """Convert a stored frame to time-sorted records; raises ValueError if it is malformed."""
    frame = result.frame
    columns = [f"prob__{label}" for label in result.classes]
    missing = [column for column in ["time", *columns] if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {', '.join(missing)}")

    probabilities = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Prediction frame contains non-finite probabilities.")
    rows = [dict(zip(result.classes, row)) for row in probabilities.tolist()]

    times = _relative_seconds(frame["time"], recording_start)
    if len(np.unique(times)) != len(times):
        raise ValueError("Prediction frame contains duplicate times.")
    order = np.argsort(times)

    if result.kind == PredictionKind.CONTINUOUS:
        return [PointPrediction(float(times[i]), rows[i]) for i in order]
    half = result.resolution / 2
    return [
        IntervalPrediction(float(times[i] - half), float(times[i] + half), max(result.classes, key=rows[i].__getitem__), rows[i])
        for i in order
    ]


def model_events(result: PredictionResult, recording_start: datetime) -> list[ModelEvent]:
    """Merge adjacent intervals with the same non-background class; IDs are stable across export and import."""
    if result.kind != PredictionKind.EVENT:
        return []
    background = background_class(result.classes)
    groups: list[list[IntervalPrediction]] = []
    for interval in map_predictions(result, recording_start):
        if interval.label == background:
            continue
        last = groups[-1][-1] if groups else None
        if last and last.label == interval.label and math.isclose(last.end, interval.start, abs_tol=1e-6):
            groups[-1].append(interval)
        else:
            groups.append([interval])
    return [
        ModelEvent(
            f"model:{result.task}:{round(group[0].start * 1000)}", group[0].start, group[-1].end, group[0].label,
            sum(interval.probabilities[interval.label] for interval in group) / len(group),
        )
        for group in groups
    ]


def downsample(records: list, max_points: int) -> list:
    """Merge consecutive records into max_points groups, keeping each class's highest probability."""
    if len(records) <= max_points:
        return records
    edges = np.linspace(0, len(records), max_points + 1).astype(int)
    reduced = []
    for group in (records[a:b] for a, b in zip(edges, edges[1:])):
        probabilities = {label: max(r.probabilities[label] for r in group) for label in group[0].probabilities}
        if isinstance(group[0], PointPrediction):
            reduced.append(PointPrediction(group[0].time, probabilities))
        else:
            reduced.append(IntervalPrediction(group[0].start, group[-1].end, max(probabilities, key=probabilities.__getitem__), probabilities))
    return reduced


def epoch_starts(result: PredictionResult, recording_start: datetime) -> list[float]:
    """Read the actual epoch timeline without mapping or changing probabilities."""
    if result.kind != PredictionKind.EPOCH:
        raise ValueError(f"Cannot map {result.kind.value} predictions as epoch.")
    return sorted((_relative_seconds(result.frame["time"], recording_start) - result.resolution / 2).tolist())


def resolve_epoch_start(start: float, duration: float, resolution: float, starts: Sequence[float]) -> float:
    """Match a requested start to a sorted epoch timeline, with 1 µs tolerance."""
    tolerance = 1e-6
    if not all(math.isfinite(value) for value in (start, duration, resolution)) or resolution <= 0 or start < 0:
        raise ValueError("Invalid epoch annotation range.")
    index = bisect_left(starts, start)
    for candidate_index in (index, index - 1):
        if 0 <= candidate_index < len(starts):
            candidate = starts[candidate_index]
            if 0 <= candidate and candidate + resolution <= duration + tolerance and math.isclose(start, candidate, rel_tol=0.0, abs_tol=tolerance):
                return candidate
    raise ValueError("Epoch start must match an existing prediction interval.")


def _relative_seconds(values: pd.Series, recording_start: datetime) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(values):
        seconds = values.to_numpy(dtype=float)
    else:
        times = pd.to_datetime(values, format="ISO8601")
        start = pd.Timestamp(recording_start)
        if times.dt.tz is None or start.tzinfo is None:
            times = times.dt.tz_localize(None)
            start = start.tz_localize(None)
        seconds = (times - start).dt.total_seconds().to_numpy()
    if not np.isfinite(seconds).all():
        raise ValueError("Prediction frame contains an invalid timestamp.")
    return seconds
