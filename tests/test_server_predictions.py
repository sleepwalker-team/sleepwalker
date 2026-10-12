from datetime import datetime

import pandas as pd
from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.services.session_service import SessionService
from sleepwalker.server.stores.prediction_store import (
    PredictionKind,
    PredictionResult,
    PredictionStore,
)
from sleepwalker.server.stores.session_store import SessionState, SessionStore


def create_prediction_app(
    *,
    task: str = "sleep_stage",
    classes: tuple[str, ...] = ("wake", "n1"),
    resolution: float = 30.0,
    frame: pd.DataFrame | None = None,
    kind: PredictionKind = PredictionKind.EPOCH,
):
    session_service = SessionService(SessionStore())
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
    }


def test_get_event_predictions_returns_explicit_intervals() -> None:
    app, session = create_prediction_app(
        task="arousal",
        classes=("no_arousal", "arousal"),
        resolution=1.0,
        frame=pd.DataFrame(
            {
                "start": [12.5, 40.0],
                "end": [18.0, 45.0],
                "prob__no_arousal": [0.08, 0.2],
                "prob__arousal": [0.92, 0.8],
            }
        ),
        kind=PredictionKind.EVENT,
    )

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/sessions/{session.id}/predictions/arousal?start=10&end=20"
        )

    assert response.status_code == 200
    assert response.json() == {
        "task": "arousal",
        "kind": "event",
        "classes": ["no_arousal", "arousal"],
        "predictions": [
            {
                "start": 12.5,
                "end": 18.0,
                "label": "arousal",
                "probabilities": {"no_arousal": 0.08, "arousal": 0.92},
            }
        ],
    }
