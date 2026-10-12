import math
import re
from dataclasses import dataclass
from datetime import datetime
from numbers import Real

import pandas as pd

from ..stores.prediction_store import PredictionKind, PredictionResult


@dataclass(frozen=True)
class ProbabilitySample:
    time: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class EpochPrediction:
    start: float
    end: float
    label: str
    probabilities: dict[str, float]


@dataclass(frozen=True)
class ContinuousPrediction:
    time: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class EventPrediction:
    start: float
    end: float
    label: str
    probabilities: dict[str, float]


def validate_prediction_result(
    result: PredictionResult,
    recording_start: datetime,
) -> None:
    if result.kind == PredictionKind.EPOCH:
        map_epoch_predictions(result, recording_start)
    elif result.kind == PredictionKind.CONTINUOUS:
        map_continuous_predictions(result, recording_start)
    else:
        map_event_predictions(result, recording_start)


def map_epoch_predictions(
    result: PredictionResult,
    recording_start: datetime,
) -> list[EpochPrediction]:
    _require_kind(result, PredictionKind.EPOCH)
    half_resolution = result.resolution / 2.0
    predictions = []

    for sample in _map_probability_samples(result, recording_start):
        label = max(result.classes, key=sample.probabilities.__getitem__)
        predictions.append(
            EpochPrediction(
                start=sample.time - half_resolution,
                end=sample.time + half_resolution,
                label=label,
                probabilities=sample.probabilities,
            )
        )

    return predictions


def map_continuous_predictions(
    result: PredictionResult,
    recording_start: datetime,
) -> list[ContinuousPrediction]:
    _require_kind(result, PredictionKind.CONTINUOUS)
    return [
        ContinuousPrediction(
            time=sample.time,
            probabilities=sample.probabilities,
        )
        for sample in _map_probability_samples(result, recording_start)
    ]


def map_event_predictions(
    result: PredictionResult,
    recording_start: datetime,
) -> list[EventPrediction]:
    _require_kind(result, PredictionKind.EVENT)
    probability_columns = {
        label: _probability_column("", label)
        for label in result.classes
    }
    required_columns = ["start", "end", *probability_columns.values()]
    missing_columns = [
        column for column in required_columns if column not in result.frame.columns
    ]

    if missing_columns:
        raise ValueError(
            "Event prediction frame is missing columns: "
            f"{', '.join(missing_columns)}"
        )

    predictions = []

    for _, row in result.frame.iterrows():
        start = _relative_seconds(row["start"], recording_start)
        end = _relative_seconds(row["end"], recording_start)

        if start < 0.0 or end <= start:
            raise ValueError("Event prediction frame contains an invalid time range.")

        probabilities = _read_probabilities(row, probability_columns)
        predictions.append(
            EventPrediction(
                start=start,
                end=end,
                label=max(result.classes, key=probabilities.__getitem__),
                probabilities=probabilities,
            )
        )

    return sorted(predictions, key=lambda prediction: (prediction.start, prediction.end))


def _map_probability_samples(
    result: PredictionResult,
    recording_start: datetime,
) -> list[ProbabilitySample]:
    grouped: dict[float, list[dict[str, float]]] = {}

    for prefix, time_column in _prediction_prefixes(result):
        probability_columns = {
            label: _probability_column(prefix, label)
            for label in result.classes
        }
        missing_columns = [
            column
            for column in probability_columns.values()
            if column not in result.frame.columns
        ]

        if missing_columns:
            raise ValueError(
                "Prediction frame is missing columns: "
                f"{', '.join(missing_columns)}"
            )

        for _, row in result.frame.iterrows():
            time = _relative_seconds(row[time_column], recording_start)
            grouped.setdefault(time, []).append(
                _read_probabilities(row, probability_columns)
            )

    return [
        ProbabilitySample(
            time=time,
            probabilities={
                label: sum(sample[label] for sample in samples) / len(samples)
                for label in result.classes
            },
        )
        for time, samples in sorted(grouped.items())
    ]


def _read_probabilities(
    row: pd.Series,
    columns: dict[str, str],
) -> dict[str, float]:
    probabilities = {
        label: float(row[column])
        for label, column in columns.items()
    }

    if any(not math.isfinite(value) for value in probabilities.values()):
        raise ValueError("Prediction frame contains non-finite probabilities.")

    return probabilities


def _require_kind(result: PredictionResult, expected: PredictionKind) -> None:
    if result.kind != expected:
        raise ValueError(
            f"Cannot map {result.kind.value} predictions as {expected.value}."
        )


def _prediction_prefixes(result: PredictionResult) -> list[tuple[str, str]]:
    task_time = f"{result.task}__time"

    if task_time in result.frame.columns:
        return [(result.task, task_time)]

    step_pattern = re.compile(rf"^{re.escape(result.task)}__step_(\d+)__time$")
    steps = []

    for column in result.frame.columns:
        match = step_pattern.match(str(column))

        if match:
            steps.append((int(match.group(1)), str(column)))

    if steps:
        return [
            (f"{result.task}__step_{step}", column)
            for step, column in sorted(steps)
        ]

    if "time" in result.frame.columns:
        return [("", "time")]

    raise ValueError("Prediction frame does not contain a supported time column.")


def _probability_column(prefix: str, label: str) -> str:
    return f"{prefix}__prob__{label}" if prefix else f"prob__{label}"


def _relative_seconds(value, recording_start: datetime) -> float:
    if isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    else:
        timestamp = pd.Timestamp(value)
        start = pd.Timestamp(recording_start)

        if timestamp.tzinfo is not None and start.tzinfo is None:
            timestamp = timestamp.tz_localize(None)
        elif timestamp.tzinfo is None and start.tzinfo is not None:
            start = start.tz_localize(None)

        result = (timestamp - start).total_seconds()

    if not math.isfinite(result):
        raise ValueError("Prediction frame contains an invalid timestamp.")

    return result
