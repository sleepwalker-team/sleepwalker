from dataclasses import replace
from datetime import datetime

import pandas as pd
import pytest

from sleepwalker.server.results.predictions import IntervalPrediction, background_class, map_predictions, model_events, epoch_starts, resolve_epoch_start
from sleepwalker.server.results.predictions import PredictionKind, PredictionResult


@pytest.mark.parametrize("absolute", [False, True])
def test_epoch_starts_follow_actual_predictions_not_a_zero_based_grid(absolute):
    times = [60.0, 40.0]
    if absolute:
        times = [pd.Timestamp("2025-01-01") + pd.Timedelta(seconds=time) for time in times]
    result = PredictionResult("sess-test", "model", "desaturation", ("negative", "positive"), 10.0, pd.DataFrame({"time": times}))
    starts = epoch_starts(result, datetime(2025, 1, 1))
    assert starts == [35.0, 55.0]
    assert resolve_epoch_start(35.0000001, 120, 10, starts) == 35.0
    with pytest.raises(ValueError):
        resolve_epoch_start(45, 120, 10, starts)


@pytest.mark.parametrize("start", [-1.0, 0.0, 30.0, 35.1, 45.0, float("nan"), float("inf")])
def test_epoch_start_resolution_rejects_invalid_or_missing_intervals(start):
    with pytest.raises(ValueError):
        resolve_epoch_start(start, 120, 10, [35.0, 55.0])


def test_epoch_start_resolution_rejects_empty_timelines_and_incomplete_epochs():
    with pytest.raises(ValueError):
        resolve_epoch_start(0, 120, 10, [])
    with pytest.raises(ValueError):
        resolve_epoch_start(115, 120, 10, [115.0])
    assert resolve_epoch_start(110, 120, 10, [110.0]) == 110.0


def test_maps_multiclass_predictions_to_relative_epochs() -> None:
    result = PredictionResult(
        session_id="sess-test",
        model_id="sleep-model",
        task="sleep_stage",
        classes=("wake", "n1"),
        resolution=30.0,
        frame=pd.DataFrame(
            {
                "time": [pd.Timestamp("2025-01-01 00:00:15"), pd.Timestamp("2025-01-01 00:00:45")],
                "prob__wake": [0.9, 0.2],
                "prob__n1": [0.1, 0.8],
            }
        ),
    )

    predictions = map_predictions(result, datetime(2025, 1, 1))

    assert [(prediction.start, prediction.end, prediction.label) for prediction in predictions] == [
        (0.0, 30.0, "wake"),
        (30.0, 60.0, "n1"),
    ]


def test_maps_continuous_predictions_without_creating_intervals() -> None:
    result = PredictionResult(
        session_id="sess-test",
        model_id="arousal-model",
        task="arousal",
        classes=("no_arousal", "arousal"),
        resolution=1.0,
        frame=pd.DataFrame(
            {
                "time": [5.0, 6.0],
                "prob__no_arousal": [0.8, 0.1],
                "prob__arousal": [0.2, 0.9],
            }
        ),
        kind=PredictionKind.CONTINUOUS,
    )

    predictions = map_predictions(result, datetime(2025, 1, 1))

    assert [prediction.time for prediction in predictions] == [5.0, 6.0]
    assert predictions[1].probabilities == {"no_arousal": 0.1, "arousal": 0.9}


def _events(classes, probabilities, resolution=1.0):
    frame = pd.DataFrame({"time": [i * resolution + resolution / 2 for i in range(len(probabilities))]})
    for index, label in enumerate(classes):
        frame[f"prob__{label}"] = [row[index] for row in probabilities]
    result = PredictionResult("sess-test", "model", "arousal", classes, resolution, frame, PredictionKind.EVENT)
    return map_predictions(result, datetime(2025, 1, 1)), model_events(result, datetime(2025, 1, 1))


def test_event_tasks_map_to_intervals_like_epochs() -> None:
    intervals, _ = _events(("no_arousal", "arousal"), [(0.08, 0.92)], resolution=5.0)
    assert intervals == [IntervalPrediction(0.0, 5.0, "arousal", {"no_arousal": 0.08, "arousal": 0.92})]


@pytest.mark.parametrize(("classes", "expected"), [
    (("no_arousal", "arousal"), "no_arousal"),
    (("desaturation", "no desaturation"), "no desaturation"),
    (("apnea", "hypopnea", "Regular breathing"), "Regular breathing"),
    (("wake", "n1", "n2", "n3", "rem"), None),
    (("none", "no_event", "event"), None),
    (("north", "notch"), None),
])
def test_background_class_needs_exactly_one_negative_class(classes, expected) -> None:
    assert background_class(classes) == expected


def test_model_events_merge_adjacent_intervals_of_the_same_class() -> None:
    _, events = _events(("apnea", "hypopnea", "regular breathing"), [
        (0.1, 0.1, 0.8), (0.6, 0.3, 0.1), (0.8, 0.1, 0.1), (0.2, 0.7, 0.1), (0.1, 0.1, 0.8), (0.1, 0.5, 0.4),
    ])
    assert [(e.id, e.start, e.end, e.label) for e in events] == [
        ("model:arousal:1000", 1.0, 3.0, "apnea"),
        ("model:arousal:3000", 3.0, 4.0, "hypopnea"),
        ("model:arousal:5000", 5.0, 6.0, "hypopnea"),
    ]
    assert events[0].confidence == pytest.approx(0.7)
    assert events[2].confidence == pytest.approx(0.5)


def test_model_events_do_not_bridge_missing_intervals() -> None:
    frame = pd.DataFrame({"time": [0.5, 2.5], "prob__no_arousal": [0.1, 0.1], "prob__arousal": [0.9, 0.9]})
    result = PredictionResult("sess-test", "model", "arousal", ("no_arousal", "arousal"), 1.0, frame, PredictionKind.EVENT)
    assert [(e.start, e.end) for e in model_events(result, datetime(2025, 1, 1))] == [(0.0, 1.0), (2.0, 3.0)]
    assert model_events(replace(result, kind=PredictionKind.EPOCH), datetime(2025, 1, 1)) == []
