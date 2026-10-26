from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sleepwalker.server.app import create_app
from sleepwalker.server.services.session_service import SessionService
from sleepwalker.server.storage.temporary_storage import TemporaryStorage
from sleepwalker.server.services.job_service import JobService
from sleepwalker.server.stores.job_store import JobState, JobStore
from sleepwalker.server.stores.prediction_store import PredictionResult, PredictionStore
from sleepwalker.server.stores.session_store import SessionStore


DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def session_api(tmp_path):
    store = SessionStore()
    service = SessionService(store)
    storage = TemporaryStorage(tmp_path / "uploads")

    with TestClient(create_app(session_service=service, temporary_storage=storage)) as client:
        yield client, store, storage


def test_session_lifecycle_for_valid_edf(session_api):
    client, store, storage = session_api
    edf_path = DATA_DIR / "signals_01.edf"

    with edf_path.open("rb") as edf_file:
        create_response = client.post(
            "/api/v1/sessions",
            files=[
                (
                    "files",
                    (edf_path.name, edf_file, "application/octet-stream"),
                )
            ],
        )

    assert create_response.status_code == 201

    created_session = create_response.json()
    session_id = created_session["id"]

    assert created_session["last_accessed_at"] == created_session["created_at"]
    assert created_session["state"] == "ready"
    assert created_session["format"] == "edf"
    assert created_session["duration"] == 3600.0
    assert created_session["recording_start"] == "2025-01-01T00:00:00"
    assert created_session["tasks"] == []
    assert created_session["signals"] == [
        {
            "id": "EEG",
            "label": "EEG",
            "unit": "uV",
            "sample_rate": 200.0,
        },
        {
            "id": "EOG",
            "label": "EOG",
            "unit": "uV",
            "sample_rate": 100.0,
        },
        {
            "id": "EMG",
            "label": "EMG",
            "unit": "uV",
            "sample_rate": 100.0,
        },
    ]
    assert (storage.root / session_id / edf_path.name).is_file()

    get_response = client.get(f"/api/v1/sessions/{session_id}")

    assert get_response.status_code == 200

    retrieved_session = get_response.json()

    assert retrieved_session["expires_at"] > created_session["expires_at"]
    assert retrieved_session["last_accessed_at"] >= created_session["last_accessed_at"]

    retrieved_session["expires_at"] = created_session["expires_at"]
    retrieved_session["last_accessed_at"] = created_session["last_accessed_at"]
    assert retrieved_session == created_session

    delete_response = client.delete(f"/api/v1/sessions/{session_id}")

    assert delete_response.status_code == 204
    assert store.get(session_id) is None
    assert not (storage.root / session_id).exists()

    missing_response = client.get(f"/api/v1/sessions/{session_id}")

    assert missing_response.status_code == 404
    assert missing_response.json() == {
        "error": {
            "code": "SESSION_NOT_FOUND",
            "message": "Session not found.",
            "details": None,
        }
    }


def test_get_session_returns_current_state(tmp_path) -> None:
    service = SessionService(SessionStore())
    session = service.create(source_files=[])
    storage = TemporaryStorage(tmp_path / "uploads")

    with TestClient(create_app(session_service=service, temporary_storage=storage)) as client:
        response = client.get(f"/api/v1/sessions/{session.id}")

    assert response.status_code == 200
    assert response.json()["state"] == "created"


def test_delete_session_removes_predictions(tmp_path) -> None:
    service = SessionService(SessionStore())
    session = service.create(source_files=[])
    storage = TemporaryStorage(tmp_path / "uploads")
    prediction_store = PredictionStore()
    prediction_store.add(
        PredictionResult(
            session_id=session.id,
            model_id="model-a",
            task="sleep_stage",
            classes=("wake", "n1"),
            resolution=30.0,
            frame=pd.DataFrame({"prediction": ["wake"]}),
        )
    )

    app = create_app(
        session_service=service,
        temporary_storage=storage,
        prediction_store=prediction_store,
    )

    with TestClient(app) as client:
        response = client.delete(f"/api/v1/sessions/{session.id}")

    assert response.status_code == 204
    assert prediction_store.list_for_session(session.id) == []


def test_delete_session_requests_active_job_cancellation(tmp_path) -> None:
    service = SessionService(SessionStore())
    session = service.create(source_files=[])
    storage = TemporaryStorage(tmp_path / "uploads")
    session_directory = storage.root / session.id
    session_directory.mkdir()
    job_service = JobService(JobStore())
    job = job_service.create(session.id, "inference")
    app = create_app(
        session_service=service,
        temporary_storage=storage,
        job_service=job_service,
    )

    with TestClient(app) as client:
        response = client.delete(f"/api/v1/sessions/{session.id}")

    assert response.status_code == 204
    assert service.store.get(session.id) is None
    assert job.state == JobState.CANCELLING
    assert session_directory.exists()

    job_service.mark_cancelled(job.id)
    app.state.session_cleanup_service.finalize_pending_deletion(session.id)

    assert job_service.get(job.id) is None
    assert not session_directory.exists()


def test_invalid_edf_removes_session_and_temporary_files(session_api):
    client, store, storage = session_api

    response = client.post(
        "/api/v1/sessions",
        files=[
            (
                "files",
                ("broken.edf", b"not an EDF file", "application/octet-stream"),
            )
        ],
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_STUDY"
    assert not store._sessions
    assert {path.name for path in storage.root.iterdir()} == {".server.lock"}


def test_file_too_large_removes_session_and_temporary_files(tmp_path) -> None:
    store = SessionStore()
    service = SessionService(store)
    storage = TemporaryStorage(tmp_path / "uploads", max_file_size_bytes=4)
    app = create_app(session_service=service, temporary_storage=storage)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sessions",
            files=[
                (
                    "files",
                    ("recording.edf", b"12345", "application/octet-stream"),
                )
            ],
        )

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "code": "FILE_TOO_LARGE",
            "message": "Uploaded file exceeds the maximum allowed size.",
            "details": {
                "filename": "recording.edf",
                "max_size_bytes": 4,
            },
        }
    }
    assert not store._sessions
    assert {path.name for path in storage.root.iterdir()} == {".server.lock"}


@pytest.mark.parametrize("filename", ["recording.nif", "recording.ndf"])
def test_native_nox_study_is_rejected_and_cleaned_up(session_api, filename):
    client, store, storage = session_api

    response = client.post(
        "/api/v1/sessions",
        files=[
            (
                "files",
                (filename, b"native Nox data", "application/octet-stream"),
            )
        ],
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_STUDY"
    assert "not supported" in response.json()["error"]["message"]
    assert not store._sessions
    assert {path.name for path in storage.root.iterdir()} == {".server.lock"}


def test_missing_files_returns_invalid_request_error(session_api):
    client, store, storage = session_api

    response = client.post("/api/v1/sessions")

    assert response.status_code == 400

    error = response.json()["error"]
    assert error["code"] == "INVALID_REQUEST"
    assert error["message"] == "Request validation failed."
    assert error["details"]["errors"]
    assert not store._sessions
    assert {path.name for path in storage.root.iterdir()} == {".server.lock"}


def test_internal_error_is_hidden_and_session_is_cleaned_up(
    tmp_path,
    monkeypatch,
):
    store = SessionStore()
    service = SessionService(store)
    storage = TemporaryStorage(tmp_path / "uploads")

    async def fail_to_save_files(*_args, **_kwargs):
        raise RuntimeError("internal implementation detail")

    monkeypatch.setattr(storage, "save_files", fail_to_save_files)

    app = create_app(session_service=service, temporary_storage=storage)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/sessions",
            files=[
                (
                    "files",
                    ("study.edf", b"EDF data", "application/octet-stream"),
                )
            ],
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An unexpected server error occurred.",
            "details": None,
        }
    }
    assert "implementation detail" not in response.text
    assert not store._sessions
    assert {path.name for path in storage.root.iterdir()} == {".server.lock"}


def test_get_session_includes_prediction_tasks(tmp_path) -> None:
    service = SessionService(SessionStore())
    session = service.create(source_files=[])
    prediction_store = PredictionStore()
    prediction_store.add(
        PredictionResult(
            session_id=session.id,
            model_id="sleep-model",
            task="sleep_stage",
            classes=("wake", "n1"),
            resolution=30.0,
            frame=pd.DataFrame({"prediction": ["wake"]}),
        )
    )

    app = create_app(
        session_service=service,
        temporary_storage=TemporaryStorage(tmp_path / "uploads"),
        prediction_store=prediction_store,
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}")

    assert response.status_code == 200
    assert response.json()["tasks"] == [
        {
            "id": "sleep_stage",
            "name": "Sleep stage",
            "kind": "epoch",
            "classes": ["wake", "n1"],
            "resolution": 30.0,
        }
    ]
