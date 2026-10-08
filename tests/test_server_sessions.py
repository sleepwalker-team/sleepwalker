from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sleepwalker.server.api import sessions
from sleepwalker.server.app import create_app
from sleepwalker.server.services.session_service import SessionService
from sleepwalker.server.storage.temporary_storage import TemporaryStorage
from sleepwalker.server.stores.session_store import SessionStore


DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def session_api(tmp_path, monkeypatch):
    store = SessionStore()
    service = SessionService(store)
    storage = TemporaryStorage(tmp_path / "uploads")

    monkeypatch.setattr(sessions, "store", store)
    monkeypatch.setattr(sessions, "service", service)
    monkeypatch.setattr(sessions, "storage", storage)

    with TestClient(create_app()) as client:
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

    assert created_session["state"] == "ready"
    assert created_session["format"] == "edf"
    assert created_session["duration"] == 3600.0
    assert created_session["recording_start"] == "2025-01-01T00:00:00"
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
    assert get_response.json() == created_session

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
    assert list(storage.root.iterdir()) == []


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
    assert list(storage.root.iterdir()) == []


def test_missing_files_returns_invalid_request_error(session_api):
    client, store, storage = session_api

    response = client.post("/api/v1/sessions")

    assert response.status_code == 400

    error = response.json()["error"]
    assert error["code"] == "INVALID_REQUEST"
    assert error["message"] == "Request validation failed."
    assert error["details"]["errors"]
    assert not store._sessions
    assert list(storage.root.iterdir()) == []


def test_internal_error_is_hidden_and_session_is_cleaned_up(
    session_api,
    monkeypatch,
):
    _client, store, storage = session_api

    async def fail_to_save_files(*_args, **_kwargs):
        raise RuntimeError("internal implementation detail")

    monkeypatch.setattr(storage, "save_files", fail_to_save_files)

    with TestClient(create_app(), raise_server_exceptions=False) as client:
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
    assert list(storage.root.iterdir()) == []
