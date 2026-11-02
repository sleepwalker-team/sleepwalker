"""Model results per session and their conversion to API-facing records.

A result's frame holds `prob__<class>` columns plus `time` (epoch: interval
center; continuous: sample time) or `start`/`end` (event).
"""

import math
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
    """One epoch or event."""

    start: float
    end: float
    label: str
    probabilities: dict[str, float]


@dataclass(frozen=True)
class PointPrediction:
    """One sample of a continuous prediction."""

    time: float
    probabilities: dict[str, float]


class PredictionStore:
    """Holds at most one result per session and task."""

    def __init__(self) -> None:
        self._results: dict[tuple[str, str], PredictionResult] = {}
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

    def list_for_session(self, session_id: str) -> list[PredictionResult]:
        with self._lock:
            results = [result for (stored_session_id, _), result in self._results.items() if stored_session_id == session_id]
        return sorted(results, key=lambda result: result.model_id)

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            keys = [key for key in self._results if key[0] == session_id]
            for key in keys:
                del self._results[key]
            return len(keys)


def map_predictions(result: PredictionResult, recording_start: datetime) -> list[IntervalPrediction | PointPrediction]:
    """Convert a stored frame to time-sorted records; raises ValueError if it is malformed."""
    frame = result.frame
    columns = [f"prob__{label}" for label in result.classes]
    required = ["start", "end", *columns] if result.kind == PredictionKind.EVENT else ["time", *columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {', '.join(missing)}")

    probabilities = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Prediction frame contains non-finite probabilities.")
    rows = [dict(zip(result.classes, row)) for row in probabilities.tolist()]

    if result.kind == PredictionKind.EVENT:
        starts = _relative_seconds(frame["start"], recording_start)
        ends = _relative_seconds(frame["end"], recording_start)
        if (starts < 0).any() or (ends <= starts).any():
            raise ValueError("Event prediction frame contains an invalid time range.")
        order = np.lexsort((ends, starts))
    else:
        times = _relative_seconds(frame["time"], recording_start)
        if len(np.unique(times)) != len(times):
            raise ValueError("Prediction frame contains duplicate times.")
        order = np.argsort(times)

    if result.kind == PredictionKind.CONTINUOUS:
        return [PointPrediction(float(times[i]), rows[i]) for i in order]
    if result.kind == PredictionKind.EPOCH:
        half = result.resolution / 2
        starts, ends = times - half, times + half
    return [
        IntervalPrediction(float(starts[i]), float(ends[i]), max(result.classes, key=rows[i].__getitem__), rows[i])
        for i in order
    ]


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
