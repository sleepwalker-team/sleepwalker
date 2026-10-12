from datetime import datetime

import pandas as pd
from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.services.session_service import SessionService
from sleepwalker.server.stores.prediction_store import PredictionResult, PredictionStore
from sleepwalker.server.stores.session_store import SessionState, SessionStore


def create_prediction_app():
    session_service = SessionService(SessionStore())
    session = session_service.create(source_files=[])
    session.state = SessionState.READY
    session.duration = 60.0
    session.recording_start = datetime(2025, 1, 1)
    prediction_store = PredictionStore()
    prediction_store.add(
        PredictionResult(
            session_id=session.id,
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
        response = client.get(f"/api/v1/sessions/{session.id}/predictions/sleep_stage?start=35&end=50")

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
        response = client.get(f"/api/v1/sessions/{session.id}/predictions/sleep_stage?start=50&end=40")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_TIME_RANGE"


def test_get_predictions_rejects_unknown_task() -> None:
    app, session = create_prediction_app()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/predictions/unknown")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TASK_NOT_FOUND"
