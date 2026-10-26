from datetime import datetime

import pandas as pd
import pytest

from sleepwalker.server.services.prediction_mapper import (
    map_continuous_predictions,
    map_epoch_predictions,
    map_event_predictions,
    epoch_starts,
    resolve_epoch_start,
)
from sleepwalker.server.stores.prediction_store import PredictionKind, PredictionResult


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

    predictions = map_epoch_predictions(result, datetime(2025, 1, 1))

    assert [(prediction.start, prediction.end, prediction.label) for prediction in predictions] == [
        (0.0, 30.0, "wake"),
        (30.0, 60.0, "n1"),
    ]


def test_maps_multitask_steps_and_averages_duplicate_times() -> None:
    result = PredictionResult(
        session_id="sess-test",
        model_id="arousal-model",
        task="arousal",
        classes=("no_arousal", "arousal"),
        resolution=10.0,
        frame=pd.DataFrame(
            {
                "arousal__step_0__time": [5.0, 5.0],
                "arousal__step_0__prob__no_arousal": [0.9, 0.3],
                "arousal__step_0__prob__arousal": [0.1, 0.7],
                "arousal__step_1__time": [15.0, 15.0],
                "arousal__step_1__prob__no_arousal": [0.2, 0.4],
                "arousal__step_1__prob__arousal": [0.8, 0.6],
            }
        ),
    )

    predictions = map_epoch_predictions(result, datetime(2025, 1, 1))

    assert len(predictions) == 2
    assert predictions[0].start == 0.0
    assert predictions[0].label == "no_arousal"
    assert predictions[0].probabilities == pytest.approx({"no_arousal": 0.6, "arousal": 0.4})
    assert predictions[1].start == 10.0
    assert predictions[1].label == "arousal"


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

    predictions = map_continuous_predictions(result, datetime(2025, 1, 1))

    assert [prediction.time for prediction in predictions] == [5.0, 6.0]
    assert predictions[1].probabilities == {"no_arousal": 0.1, "arousal": 0.9}


def test_maps_event_predictions_with_explicit_boundaries() -> None:
    result = PredictionResult(
        session_id="sess-test",
        model_id="arousal-model",
        task="arousal",
        classes=("no_arousal", "arousal"),
        resolution=1.0,
        frame=pd.DataFrame(
            {
                "start": [12.5],
                "end": [18.0],
                "prob__no_arousal": [0.08],
                "prob__arousal": [0.92],
            }
        ),
        kind=PredictionKind.EVENT,
    )

    predictions = map_event_predictions(result, datetime(2025, 1, 1))

    assert len(predictions) == 1
    assert predictions[0].start == 12.5
    assert predictions[0].end == 18.0
    assert predictions[0].label == "arousal"
    assert predictions[0].probabilities == {
        "no_arousal": 0.08,
        "arousal": 0.92,
    }
