from datetime import datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.core.sessions import SessionService
from sleepwalker.server.results.predictions import PredictionKind, PredictionResult, PredictionStore
from sleepwalker.server.core.sessions import SessionState


def create_prediction_app(
    *,
    task: str = "sleep_stage",
    classes: tuple[str, ...] = ("wake", "n1"),
    resolution: float = 30.0,
    frame: pd.DataFrame | None = None,
    kind: PredictionKind = PredictionKind.EPOCH,
):
    session_service = SessionService()
    session = session_service.create(source_files=[])
    session.state = SessionState.READY
    session.duration = 60.0
    session.recording_start = datetime(2025, 1, 1)
    prediction_store = PredictionStore()
    prediction_frame = frame if frame is not None else pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2025-01-01 00:00:15"),
                pd.Timestamp("2025-01-01 00:00:45"),
            ],
            "prob__wake": [0.9, 0.2],
            "prob__n1": [0.1, 0.8],
        }
    )
    prediction_store.add(
        PredictionResult(
            session_id=session.id,
            model_id="sleep-model",
            task=task,
            classes=classes,
            resolution=resolution,
            frame=prediction_frame,
            kind=kind,
        )
    )
    app = create_app(session_service=session_service, prediction_store=prediction_store)
    return app, session


def test_list_tasks_returns_available_prediction_tasks() -> None:
    app, session = create_prediction_app()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/tasks")

    assert response.status_code == 200
    assert response.json() == {
        "tasks": [
            {
                "id": "sleep_stage",
                "name": "Sleep stage",
                "kind": "epoch",
                "classes": ["wake", "n1"],
                "background_class": None,
                "resolution": 30.0,
            }
        ]
    }


def test_get_predictions_filters_requested_time_range() -> None:
    app, session = create_prediction_app()

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/sessions/{session.id}/predictions/sleep_stage?start=35&end=50"
        )

    assert response.status_code == 200
    assert response.json() == {
        "task": "sleep_stage",
        "kind": "epoch",
        "classes": ["wake", "n1"],
        "predictions": [
            {
                "start": 30.0,
                "end": 60.0,
                "label": "n1",
                "probabilities": {"wake": 0.2, "n1": 0.8},
            }
        ],
        "downsampled": False,
    }


def test_get_predictions_rejects_invalid_time_range() -> None:
    app, session = create_prediction_app()

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/sessions/{session.id}/predictions/sleep_stage?start=50&end=40"
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_TIME_RANGE"


def test_get_predictions_rejects_unknown_task() -> None:
    app, session = create_prediction_app()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/predictions/unknown")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TASK_NOT_FOUND"


def test_get_continuous_predictions_returns_probability_samples() -> None:
    app, session = create_prediction_app(
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

    with TestClient(app) as client:
        tasks_response = client.get(f"/api/v1/sessions/{session.id}/tasks")
        response = client.get(
            f"/api/v1/sessions/{session.id}/predictions/arousal?start=5.5&end=7"
        )

    assert tasks_response.status_code == 200
    assert tasks_response.json()["tasks"][0]["kind"] == "continuous"
    assert tasks_response.json()["tasks"][0]["background_class"] == "no_arousal"
    assert response.status_code == 200
    assert response.json() == {
        "task": "arousal",
        "kind": "continuous",
        "classes": ["no_arousal", "arousal"],
        "predictions": [
            {
                "time": 6.0,
                "probabilities": {"no_arousal": 0.1, "arousal": 0.9},
            }
        ],
        "downsampled": False,
    }


def test_get_event_predictions_returns_intervals_and_events() -> None:
    app, session = create_prediction_app(
        task="arousal",
        classes=("no_arousal", "arousal"),
        resolution=5.0,
        frame=pd.DataFrame({"time": [7.5, 12.5, 17.5, 42.5], "prob__no_arousal": [0.9, 0.2, 0.4, 0.1], "prob__arousal": [0.1, 0.8, 0.6, 0.9]}),
        kind=PredictionKind.EVENT,
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/predictions/arousal?start=11&end=20")

    assert response.status_code == 200
    body = response.json()
    assert [(p["start"], p["label"]) for p in body["predictions"]] == [(10.0, "arousal"), (15.0, "arousal")]
    assert body["events"] == [{"id": "model:arousal:10000", "start": 10.0, "end": 20.0, "label": "arousal", "confidence": pytest.approx(0.7)}]


def test_get_context_returns_prediction_of_every_task_at_time() -> None:
    app, session = create_prediction_app()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/context?time=35")

    assert response.status_code == 200
    assert response.json() == {
        "time": 35.0,
        "tasks": {"sleep_stage": {"prediction": "n1", "confidence": 0.8, "probabilities": {"wake": 0.2, "n1": 0.8}}},
    }


def test_get_context_uses_nearest_continuous_sample() -> None:
    app, session = create_prediction_app(
        task="arousal",
        classes=("no_arousal", "arousal"),
        resolution=1.0,
        frame=pd.DataFrame({"time": [5.0, 6.0], "prob__no_arousal": [0.8, 0.1], "prob__arousal": [0.2, 0.9]}),
        kind=PredictionKind.CONTINUOUS,
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/context?time=5.4")

    assert response.json()["tasks"]["arousal"] == {"prediction": "no_arousal", "confidence": 0.8, "probabilities": {"no_arousal": 0.8, "arousal": 0.2}}


def test_get_context_returns_null_for_gaps() -> None:
    app, session = create_prediction_app(
        frame=pd.DataFrame({"time": [pd.Timestamp("2025-01-01 00:00:15")], "prob__wake": [0.9], "prob__n1": [0.1]}),
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/context?time=45")

    assert response.json() == {"time": 45.0, "tasks": {"sleep_stage": None}}


@pytest.mark.parametrize("time", [-1, 61])
def test_get_context_rejects_time_outside_recording(time) -> None:
    app, session = create_prediction_app()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/context?time={time}")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_TIME_RANGE"


def test_get_predictions_applies_label_corrections_to_effective() -> None:
    app, session = create_prediction_app()
    app.state.annotation_service.set_label(session.id, "sleep_stage", 30.0, "wake")

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/predictions/sleep_stage?start=10&end=60&effective=true")

    assert response.json()["effective"] == [
        {"start": 0.0, "end": 30.0, "label": "wake", "source": "model", "confidence": 0.9},
        {"start": 30.0, "end": 60.0, "label": "wake", "source": "human"},
    ]


def test_get_predictions_applies_event_edits_to_effective() -> None:
    app, session = create_prediction_app(
        task="arousal",
        classes=("no_arousal", "arousal"),
        resolution=5.0,
        frame=pd.DataFrame({"time": [7.5, 12.5, 42.5], "prob__no_arousal": [0.1, 0.9, 0.2], "prob__arousal": [0.9, 0.1, 0.8]}),
        kind=PredictionKind.EVENT,
    )
    annotations = app.state.annotation_service
    annotations.delete_event(session.id, "arousal", "model:arousal:5000")
    added = annotations.add_event(session.id, "arousal", 20.0, 25.0, "arousal")

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/predictions/arousal?effective=true")

    assert response.json()["effective"] == [
        {"start": 20.0, "end": 25.0, "label": "arousal", "source": "human", "id": added.event_id},
        {"start": 40.0, "end": 45.0, "label": "arousal", "source": "model", "confidence": 0.8, "id": "model:arousal:40000"},
    ]


def test_get_predictions_merges_intervals_to_max_points_keeping_peaks() -> None:
    app, session = create_prediction_app(
        task="arousal",
        classes=("no_arousal", "arousal"),
        resolution=5.0,
        frame=pd.DataFrame({
            "time": [2.5, 7.5, 12.5, 17.5, 22.5, 27.5],
            "prob__no_arousal": [0.9, 0.3, 0.8, 0.9, 0.9, 0.6],
            "prob__arousal": [0.1, 0.7, 0.2, 0.1, 0.1, 0.4],
        }),
        kind=PredictionKind.EVENT,
    )

    with TestClient(app) as client:
        body = client.get(f"/api/v1/sessions/{session.id}/predictions/arousal?max_points=2").json()

    assert body["downsampled"] is True
    assert body["predictions"] == [
        {"start": 0.0, "end": 15.0, "label": "no_arousal", "probabilities": {"no_arousal": 0.9, "arousal": 0.7}},
        {"start": 15.0, "end": 30.0, "label": "no_arousal", "probabilities": {"no_arousal": 0.9, "arousal": 0.4}},
    ]
    assert body["events"] == [{"id": "model:arousal:5000", "start": 5.0, "end": 10.0, "label": "arousal", "confidence": 0.7}]


def test_get_predictions_merges_continuous_samples_to_max_points() -> None:
    app, session = create_prediction_app(
        task="arousal",
        classes=("no_arousal", "arousal"),
        resolution=1.0,
        frame=pd.DataFrame({"time": [1.0, 2.0, 3.0, 4.0], "prob__no_arousal": [0.8, 0.1, 0.9, 0.7], "prob__arousal": [0.2, 0.9, 0.1, 0.3]}),
        kind=PredictionKind.CONTINUOUS,
    )

    with TestClient(app) as client:
        body = client.get(f"/api/v1/sessions/{session.id}/predictions/arousal?max_points=2").json()

    assert body["downsampled"] is True
    assert body["predictions"] == [
        {"time": 1.0, "probabilities": {"no_arousal": 0.8, "arousal": 0.9}},
        {"time": 3.0, "probabilities": {"no_arousal": 0.9, "arousal": 0.3}},
    ]


def test_get_predictions_rejects_invalid_max_points() -> None:
    app, session = create_prediction_app()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/predictions/sleep_stage?max_points=1")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_MAX_POINTS"
