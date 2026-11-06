from dataclasses import replace
from datetime import datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.results.annotations import AnnotationService
from sleepwalker.server.core.sessions import SessionService
from sleepwalker.server.results.predictions import PredictionKind, PredictionResult, PredictionStore
from sleepwalker.server.core.sessions import SessionState


def create_annotation_app(
    *,
    task: str = "sleep_stage",
    classes: tuple[str, ...] = ("wake", "n1", "n2", "n3", "rem"),
    kind: PredictionKind = PredictionKind.EPOCH,
    resolution: float = 30.0,
    starts: tuple[float, ...] = (0.0, 30.0, 60.0, 90.0),
):
    session_service = SessionService()
    session = session_service.create(source_files=[])
    session.state = SessionState.READY
    session.duration = 120.0
    session.recording_start = datetime(2025, 1, 1)
    prediction_store = PredictionStore()
    prediction_store.add(
        PredictionResult(
            session_id=session.id,
            model_id=f"{task}-model",
            task=task,
            classes=classes,
            resolution=resolution,
            frame=pd.DataFrame({
                "time": [start + resolution / 2 for start in starts],
                **{f"prob__{label}": [float(index == 0)] * len(starts) for index, label in enumerate(classes)},
            }) if kind != PredictionKind.CONTINUOUS else pd.DataFrame(),
            kind=kind,
        )
    )
    annotation_service = AnnotationService()
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
    assert annotation_service.snapshot(session.id).labels[0].label == "n3"


@pytest.mark.parametrize("start", [35.0, 35.0000001, 34.9999999, 55.0])
def test_set_label_accepts_actual_shifted_epochs_and_stores_canonical_start(start):
    app, session, annotations = create_annotation_app(task="desaturation", classes=("desaturation", "no desaturation"), resolution=10, starts=(35.0, 55.0))
    original = app.state.prediction_store.get_by_task(session.id, "desaturation").frame.copy()
    with TestClient(app) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/annotations", json={"operation": "set_label", "task": "desaturation", "start": start, "label": "desaturation"})
    assert response.status_code == 201, response.text
    expected = 35.0 if start < 40 else 55.0
    assert response.json()["start"] == expected
    assert annotations.snapshot(session.id).labels[0].start == expected
    pd.testing.assert_frame_equal(app.state.prediction_store.get_by_task(session.id, "desaturation").frame, original)


@pytest.mark.parametrize("start", [0.0, 30.0, 35.1, 40.0, 45.0, 65.0, 115.0])
def test_set_label_rejects_nonexistent_epochs_including_context_and_prediction_gaps(start):
    app, session, annotations = create_annotation_app(task="desaturation", classes=("desaturation", "no desaturation"), resolution=10, starts=(35.0, 55.0))
    with TestClient(app) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/annotations", json={"operation": "set_label", "task": "desaturation", "start": start, "label": "desaturation"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    assert annotations.snapshot(session.id).labels == []


def test_set_label_cannot_create_overrides_for_an_empty_prediction_task():
    app, session, annotations = create_annotation_app(starts=())
    with TestClient(app) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/annotations", json={"operation": "set_label", "task": "sleep_stage", "start": 0.0, "label": "n3"})
    assert response.status_code == 400
    assert annotations.snapshot(session.id).labels == []


def test_roundoff_does_not_create_separate_annotation_positions_and_revert_still_works():
    app, session, annotations = create_annotation_app(resolution=10, starts=(35.0,))
    with TestClient(app) as client:
        for start, label in [(35.0, "n3"), (35.0000001, "wake")]:
            response = client.post(f"/api/v1/sessions/{session.id}/annotations", json={"operation": "set_label", "task": "sleep_stage", "start": start, "label": label})
            assert response.status_code == 201
        assert len(annotations.snapshot(session.id).labels) == 1
        assert annotations.snapshot(session.id).labels[0].label == "wake"
        reverted = client.post(f"/api/v1/sessions/{session.id}/annotations", json={"operation": "revert", "edit_id": response.json()["edit_id"]})
    assert reverted.status_code == 201
    assert annotations.snapshot(session.id).labels[0].label == "n3"


@pytest.mark.parametrize(
    "kind",
    [PredictionKind.CONTINUOUS, PredictionKind.EVENT],
)
def test_event_can_be_added_deleted_and_delete_reverted(
    kind: PredictionKind,
) -> None:
    app, session, annotation_service = create_annotation_app(
        task="arousal",
        classes=("no_arousal", "arousal"),
        kind=kind,
    )

    with TestClient(app) as client:
        added = client.post(
            f"/api/v1/sessions/{session.id}/annotations",
            json={
                "operation": "add_event",
                "task": "arousal",
                "start": 10.0,
                "end": 20.0,
                "label": "arousal",
            },
        ).json()
        deleted = client.post(
            f"/api/v1/sessions/{session.id}/annotations",
            json={
                "operation": "delete_event",
                "task": "arousal",
                "event_id": added["event_id"],
            },
        ).json()
        reverted = client.post(
            f"/api/v1/sessions/{session.id}/annotations",
            json={"operation": "revert", "edit_id": deleted["edit_id"]},
        )

    assert reverted.status_code == 201
    assert reverted.json()["reverted_edit_id"] == deleted["edit_id"]
    assert annotation_service.snapshot(session.id).events[0].event_id == added["event_id"]


def test_annotation_rejects_invalid_label() -> None:
    app, session, _annotation_service = create_annotation_app()

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/sessions/{session.id}/annotations",
            json={
                "operation": "set_label",
                "task": "sleep_stage",
                "start": 30.0,
                "label": "invalid",
            },
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


def test_set_label_rejects_continuous_task() -> None:
    app, session, _annotation_service = create_annotation_app(
        task="arousal",
        classes=("no_arousal", "arousal"),
        kind=PredictionKind.CONTINUOUS,
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/sessions/{session.id}/annotations",
            json={
                "operation": "set_label",
                "task": "arousal",
                "start": 30.0,
                "label": "arousal",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "INVALID_REQUEST",
        "message": "Annotation operation is incompatible with the task kind.",
        "details": {
            "operation": "set_label",
            "task_kind": "continuous",
        },
    }


def test_add_event_rejects_epoch_task() -> None:
    app, session, _annotation_service = create_annotation_app()

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/sessions/{session.id}/annotations",
            json={
                "operation": "add_event",
                "task": "sleep_stage",
                "start": 10.0,
                "end": 20.0,
                "label": "n1",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "INVALID_REQUEST",
        "message": "Annotation operation is incompatible with the task kind.",
        "details": {
            "operation": "add_event",
            "task_kind": "epoch",
        },
    }


def test_get_annotations_returns_empty_lists_for_unannotated_session() -> None:
    app, session, _service = create_annotation_app()
    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/annotations")

    assert response.status_code == 200
    assert response.json() == {"edits": [], "labels": [], "events": [], "deleted_events": []}


def test_get_annotations_resolves_label_history_without_changing_model_predictions() -> None:
    app, session, _service = create_annotation_app()
    prediction_store = app.state.prediction_store
    result = prediction_store.get_by_task(session.id, "sleep_stage")
    prediction_store.delete_session(session.id)
    prediction_store.add(replace(
        result,
        frame=pd.DataFrame({
            "time": [45.0], "prob__wake": [0.05], "prob__n1": [0.05],
            "prob__n2": [0.8], "prob__n3": [0.05], "prob__rem": [0.05],
        }),
    ))
    url = f"/api/v1/sessions/{session.id}/annotations"

    with TestClient(app) as client:
        original_predictions = client.get(f"/api/v1/sessions/{session.id}/predictions/sleep_stage").json()
        first = client.post(url, json={
            "operation": "set_label", "task": "sleep_stage", "start": 30.0, "label": "n1",
        }).json()
        second = client.post(url, json={
            "operation": "set_label", "task": "sleep_stage", "start": 30.0, "label": "n3",
        }).json()
        before_revert = client.get(url).json()
        assert before_revert["edits"] == [first, second]
        assert before_revert["labels"] == [second]
        assert before_revert["events"] == []

        revert_second = client.post(url, json={"operation": "revert", "edit_id": second["edit_id"]}).json()
        after_revert = client.get(url).json()
        assert after_revert["edits"] == [first, second, revert_second]
        assert after_revert["labels"] == [first]

        revert_first = client.post(url, json={"operation": "revert", "edit_id": first["edit_id"]}).json()
        fully_reverted = client.get(url).json()
        assert fully_reverted["labels"] == []
        assert fully_reverted["edits"] == [first, second, revert_second, revert_first]
        after_predictions = client.get(f"/api/v1/sessions/{session.id}/predictions/sleep_stage").json()
        assert after_predictions == original_predictions
        assert original_predictions["predictions"][0]["label"] == "n2"


@pytest.mark.parametrize("kind", [PredictionKind.CONTINUOUS, PredictionKind.EVENT])
def test_get_annotations_resolves_event_deletions_and_reverts(kind) -> None:
    app, session, _service = create_annotation_app(
        task="arousal", classes=("no_arousal", "arousal"), kind=kind,
    )
    url = f"/api/v1/sessions/{session.id}/annotations"
    with TestClient(app) as client:
        first = client.post(url, json={
            "operation": "add_event", "task": "arousal", "start": 10.0, "end": 20.0, "label": "arousal",
        }).json()
        second = client.post(url, json={
            "operation": "add_event", "task": "arousal", "start": 30.0, "end": 40.0, "label": "arousal",
        }).json()
        deleted = client.post(url, json={
            "operation": "delete_event", "task": "arousal", "event_id": first["event_id"],
        }).json()
        after_delete = client.get(url).json()
        assert after_delete["events"] == [second]
        assert after_delete["labels"] == []
        assert after_delete["edits"] == [first, second, deleted]

        revert_delete = client.post(url, json={"operation": "revert", "edit_id": deleted["edit_id"]}).json()
        assert client.get(url).json()["events"] == [first, second]
        revert_second = client.post(url, json={"operation": "revert", "edit_id": second["edit_id"]}).json()
        after_reverts = client.get(url).json()
        assert after_reverts["events"] == [first]
        assert after_reverts["edits"] == [first, second, deleted, revert_delete, revert_second]


def test_get_annotations_only_returns_requested_session_data() -> None:
    app, session, service = create_annotation_app()
    local = service.set_label(session.id, "sleep_stage", 0.0, "n1")
    other = app.state.session_service.create([])
    other.state = SessionState.READY
    foreign_label = service.set_label(other.id, "sleep_stage", 0.0, "n3")
    foreign_event = service.add_event(other.id, "arousal", 1.0, 2.0, "arousal")

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/annotations")

    assert response.status_code == 200
    body = response.json()
    assert [edit["edit_id"] for edit in body["edits"]] == [local.id]
    assert body["labels"] == body["edits"]
    assert body["events"] == []
    assert foreign_label.id not in response.text
    assert foreign_event.id not in response.text


@pytest.mark.parametrize("state", [SessionState.CREATED, SessionState.PROCESSING, SessionState.FAILED])
def test_get_annotations_rejects_session_that_is_not_ready(state) -> None:
    app, session, _service = create_annotation_app()
    session.state = state
    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/annotations")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INVALID_SESSION_STATE"
    assert response.json()["error"]["details"] == {"state": state.value}


def test_get_annotations_rejects_unknown_session() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/sessions/sess-unknown/annotations")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SESSION_NOT_FOUND"


def test_get_annotations_rejects_deleted_session_and_removes_history() -> None:
    app, session, service = create_annotation_app()
    service.set_label(session.id, "sleep_stage", 0.0, "n1")
    with TestClient(app) as client:
        assert client.delete(f"/api/v1/sessions/{session.id}").status_code == 204
        response = client.get(f"/api/v1/sessions/{session.id}/annotations")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SESSION_NOT_FOUND"
    assert service.list_edits(session.id) == []


def test_model_event_can_be_deleted_once_and_restored() -> None:
    # Every interval predicts "arousal" (index 0), so the whole recording is one model event.
    app, session, annotations = create_annotation_app(task="arousal", classes=("arousal", "no_arousal"), kind=PredictionKind.EVENT)
    url = f"/api/v1/sessions/{session.id}/annotations"
    with TestClient(app) as client:
        event_id = client.get(f"/api/v1/sessions/{session.id}/predictions/arousal").json()["events"][0]["id"]
        unknown = client.post(url, json={"operation": "delete_event", "task": "arousal", "event_id": "model:arousal:5000"})
        deleted = client.post(url, json={"operation": "delete_event", "task": "arousal", "event_id": event_id})
        again = client.post(url, json={"operation": "delete_event", "task": "arousal", "event_id": event_id})
        listed = client.get(url).json()["deleted_events"]
        client.post(url, json={"operation": "revert", "edit_id": deleted.json()["edit_id"]})

    assert event_id == "model:arousal:0"
    assert unknown.status_code == again.status_code == 404
    assert deleted.status_code == 201
    assert [edit["event_id"] for edit in listed] == [event_id]
    assert annotations.snapshot(session.id).deleted_events == []
