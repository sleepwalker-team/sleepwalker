"""Pure transformations for long-form prediction tables."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def apply_pipeline(frame: pd.DataFrame, pipeline: list[Any], *, patient: str, signals: Sequence[str]) -> pd.DataFrame:
    current = frame.copy()
    for transform in pipeline:
        current = transform(current, patient=patient, signals=signals)
        if not isinstance(current, pd.DataFrame):
            raise TypeError(f"Prediction transform {transform} returned {type(current).__name__}, expected DataFrame.")
        if current.empty:
            break
    return current


def filter_patient_signals(frame: pd.DataFrame, patient: str, signals: Sequence[str], *, include_all: Sequence[str] | None = None, include_any: Sequence[str] | None = None, exclude_any: Sequence[str] | None = None) -> pd.DataFrame:
    del patient
    available = set(signals)
    keep = (not include_all or set(include_all).issubset(available)) and (not include_any or bool(set(include_any).intersection(available))) and (not exclude_any or not set(exclude_any).intersection(available))
    return frame if keep else frame.iloc[0:0]


def filter_annotation(frame: pd.DataFrame, patient: str, signals: Sequence[str], *, labels: Sequence[str], minimum_coverage: float = 0.5) -> pd.DataFrame:
    del patient, signals
    columns = [f"annotation__{label}" for label in labels]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Annotation filter refers to unavailable labels: {missing}.")
    coverage = frame[columns].sum(axis=1).clip(upper=1.0)
    return frame.loc[coverage >= float(minimum_coverage)]


def resolve_overlaps(frame: pd.DataFrame, patient: str, signals: Sequence[str], *, method: str = "mean_probability") -> pd.DataFrame:
    del patient, signals
    if method not in {"mean_probability", "majority_vote"}:
        raise ValueError(f"Unknown overlap method '{method}'.")
    probability_columns = [column for column in frame if column.startswith("prob__")]
    annotation_columns = [column for column in frame if column.startswith("annotation__")]
    if not probability_columns:
        raise ValueError("Overlap resolution requires probability columns.")
    if int(frame.groupby("time")["target"].nunique().max()) != 1:
        raise ValueError("Overlapping predictions disagree on the target at one timestamp.")
    work = frame.copy()
    if method == "majority_vote":
        votes = work[probability_columns].to_numpy().argmax(axis=1)
        for class_index, column in enumerate(probability_columns):
            work[column] = (votes == class_index).astype(float)
    aggregation = {"target": "first", **{column: "mean" for column in probability_columns + annotation_columns}}
    return work.groupby("time", sort=True, as_index=False).agg(aggregation)


def smooth_probabilities(frame: pd.DataFrame, patient: str, signals: Sequence[str], *, window: int = 3) -> pd.DataFrame:
    del patient, signals
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("Smoothing window must be a positive odd integer.")
    if window == 1 or len(frame) < 2:
        return frame
    probability_columns = [column for column in frame if column.startswith("prob__")]
    if not probability_columns:
        raise ValueError("Probability smoothing requires probability columns.")
    work = frame.sort_values("time").copy()
    differences = work["time"].diff()
    positive = differences[differences > pd.Timedelta(0)]
    expected = positive.min() if not positive.empty else None
    segments = pd.Series(0, index=work.index) if expected is None else (differences > expected * 1.5).cumsum()
    work[probability_columns] = work.groupby(segments, sort=False)[probability_columns].transform(lambda values: values.rolling(window, center=True, min_periods=1).mean())
    probabilities = work[probability_columns].to_numpy(dtype=float)
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Smoothed probabilities must have a positive row sum.")
    work[probability_columns] = probabilities / totals
    return work
