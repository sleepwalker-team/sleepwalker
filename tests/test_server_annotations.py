from datetime import datetime

import pandas as pd
from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.services.annotation_service import AnnotationService
from sleepwalker.server.services.session_service import SessionService
from sleepwalker.server.stores.annotation_store import AnnotationStore
from sleepwalker.server.stores.prediction_store import PredictionResult, PredictionStore
from sleepwalker.server.stores.session_store import SessionState, SessionStore


def create_annotation_app():
    session_service = SessionService(SessionStore())
    session = session_service.create(source_files=[])
    session.state = SessionState.READY
    session.duration = 120.0
    session.recording_start = datetime(2025, 1, 1)
    prediction_store = PredictionStore()
    prediction_store.add(
        PredictionResult(
            session_id=session.id,
            model_id="sleep-model",
            task="sleep_stage",
            classes=("wake", "n1", "n2", "n3", "rem"),
            resolution=30.0,
            frame=pd.DataFrame(),
        )
    )
    annotation_service = AnnotationService(AnnotationStore())
    app = create_app(
        session_service=session_service,
        prediction_store=prediction_store,
        annotation_service=annotation_service,
    )
    return app, session, annotation_service


def test_set_label_creates_separate_annotation_edit() -> None:
    app, session, annotation_service = create_annotation_app()

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/sessions/{session.id}/annotations",
            json={"operation": "set_label", "task": "sleep_stage", "start": 30.0, "label": "n3"},
        )

    assert response.status_code == 201
    assert response.json()["operation"] == "set_label"
    assert response.json()["edit_id"].startswith("edit_")
    assert annotation_service.current_labels(session.id)[0].label == "n3"


def test_event_can_be_added_deleted_and_delete_reverted() -> None:
    app, session, annotation_service = create_annotation_app()

    with TestClient(app) as client:
        added = client.post(
            f"/api/v1/sessions/{session.id}/annotations",
            json={
                "operation": "add_event",
                "task": "sleep_stage",
                "start": 10.0,
                "end": 20.0,
                "label": "n1",
            },
        ).json()
        deleted = client.post(
            f"/api/v1/sessions/{session.id}/annotations",
            json={"operation": "delete_event", "task": "sleep_stage", "event_id": added["event_id"]},
        ).json()
        reverted = client.post(
            f"/api/v1/sessions/{session.id}/annotations",
            json={"operation": "revert", "edit_id": deleted["edit_id"]},
        )

    assert reverted.status_code == 201
    assert reverted.json()["reverted_edit_id"] == deleted["edit_id"]
    assert annotation_service.current_events(session.id)[0].event_id == added["event_id"]


def test_annotation_rejects_invalid_label() -> None:
    app, session, _annotation_service = create_annotation_app()

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/sessions/{session.id}/annotations",
            json={"operation": "set_label", "task": "sleep_stage", "start": 30.0, "label": "invalid"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_annotation_rejects_unknown_task() -> None:
    app, session, _annotation_service = create_annotation_app()

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/sessions/{session.id}/annotations",
            json={"operation": "set_label", "task": "unknown", "start": 30.0, "label": "n3"},
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TASK_NOT_FOUND"
