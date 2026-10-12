import math
import re
from dataclasses import dataclass
from datetime import datetime
from numbers import Real

import pandas as pd

from ..stores.prediction_store import PredictionResult


@dataclass(frozen=True)
class EpochPrediction:
    start: float
    end: float
    label: str
    probabilities: dict[str, float]


def map_epoch_predictions(
    result: PredictionResult,
    recording_start: datetime,
) -> list[EpochPrediction]:
    prefixes = _prediction_prefixes(result)
    grouped: dict[float, list[dict[str, float]]] = {}

    for prefix, time_column in prefixes:
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
            center = _relative_seconds(row[time_column], recording_start)
            probabilities = {
                label: float(row[column])
                for label, column in probability_columns.items()
            }

            if any(not math.isfinite(value) for value in probabilities.values()):
                raise ValueError("Prediction frame contains non-finite probabilities.")

            grouped.setdefault(center, []).append(probabilities)

    half_resolution = result.resolution / 2.0
    predictions = []

    for center, samples in sorted(grouped.items()):
        probabilities = {
            label: sum(sample[label] for sample in samples) / len(samples)
            for label in result.classes
        }
        label = max(result.classes, key=probabilities.__getitem__)
        predictions.append(
            EpochPrediction(
                start=center - half_resolution,
                end=center + half_resolution,
                label=label,
                probabilities=probabilities,
            )
        )

    return predictions


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
