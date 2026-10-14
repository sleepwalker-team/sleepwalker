import json
from datetime import datetime, timezone
from io import BytesIO
from zipfile import ZipFile

import pytest

from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.services.job_service import JobService
from sleepwalker.server.services.session_service import SessionService
from sleepwalker.server.storage.temporary_storage import TemporaryStorage
from sleepwalker.server.stores.export_store import Export, ExportState, ExportStore
from sleepwalker.server.stores.job_store import JobStore
from sleepwalker.server.stores.session_store import SessionState, SessionStore


def _ready_session(service: SessionService):
    session = service.create(source_files=[])
    session.state = SessionState.READY
    session.duration = 60.0
    session.recording_start = datetime(2025, 1, 1)
    return session


def test_result_export_lifecycle_and_download(tmp_path) -> None:
    session_service = SessionService(SessionStore())
    session = _ready_session(session_service)
    export_store = ExportStore()
    app = create_app(
        session_service=session_service,
        export_store=export_store,
        temporary_storage=TemporaryStorage(tmp_path / "uploads"),
    )

    with TestClient(app) as client:
        create_response = client.post(
            f"/api/v1/sessions/{session.id}/exports",
            json={
                "format": "json",
                "mode": "results",
                "include": {
                    "predictions": True,
                    "probabilities": True,
                    "annotations": True,
                    "metadata": True,
                },
            },
        )

        assert create_response.status_code == 202
        assert create_response.json()["state"] == "queued"
        export_id = create_response.json()["export_id"]

        status_response = client.get(f"/api/v1/exports/{export_id}")
        download_response = client.get(f"/api/v1/exports/{export_id}/download")

    assert status_response.status_code == 200
    assert status_response.json()["state"] == "ready"
    assert status_response.json()["size"] > 0
    assert download_response.status_code == 200
    assert download_response.headers["content-type"] == "application/json"
    assert download_response.json()["format"] == "sleepwalker-results"
    assert download_response.json()["session"]["id"] == session.id
    assert download_response.json()["tasks"] == []
    assert download_response.json()["annotations"] == []


def test_export_rejects_unknown_session() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/sessions/sess-unknown/exports",
            json={"format": "json", "mode": "results"},
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SESSION_NOT_FOUND"


def test_export_rejects_second_active_session_job() -> None:
    session_service = SessionService(SessionStore())
    session = _ready_session(session_service)
    job_service = JobService(JobStore())
    active_job = job_service.create(session.id, "inference")
    app = create_app(session_service=session_service, job_service=job_service)

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/sessions/{session.id}/exports",
            json={"format": "json", "mode": "results"},
        )

    assert response.status_code == 409
    assert response.json()["error"] == {
        "code": "JOB_ALREADY_RUNNING",
        "message": "Session already has an active job.",
        "details": {"job_id": active_job.id, "state": "queued"},
    }


def test_export_status_rejects_unknown_export() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/exports/export-unknown")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "EXPORT_NOT_FOUND"


def test_download_rejects_export_that_is_not_ready() -> None:
    export_store = ExportStore()
    now = datetime.now(timezone.utc)
    export_store.add(
        Export(
            id="export-test",
            session_id="sess-test",
            job_id="job-test",
            state=ExportState.QUEUED,
            format="json",
            mode="results",
            created_at=now,
            expires_at=now,
        )
    )

    with TestClient(create_app(export_store=export_store)) as client:
        response = client.get("/api/v1/exports/export-test/download")

    assert response.status_code == 409
    assert response.json()["error"] == {
        "code": "EXPORT_NOT_READY",
        "message": "Export is not ready for download.",
        "details": {"state": "queued"},
    }


def test_workspace_download_and_session_delete(tmp_path) -> None:
    session_service = SessionService(SessionStore())
    session = _ready_session(session_service)
    source = tmp_path / "record.edf"
    source.write_bytes(b"test recording")
    session.source_files = [source]
    storage = TemporaryStorage(tmp_path / "uploads")
    app = create_app(session_service=session_service, temporary_storage=storage)

    with TestClient(app) as client:
        created = client.post(
            f"/api/v1/sessions/{session.id}/exports",
            json={"format": "sleepwalker", "mode": "workspace"},
        )
        assert created.status_code == 202
        export_id = created.json()["export_id"]
        export_status = client.get(f"/api/v1/exports/{export_id}")
        assert export_status.json()["state"] == "ready"
        assert export_status.json()["format"] == "sleepwalker"
        assert export_status.json()["mode"] == "workspace"
        download = client.get(f"/api/v1/exports/{export_id}/download")
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/zip"
        assert ".sleepwalker.zip" in download.headers["content-disposition"]
        with ZipFile(BytesIO(download.content)) as archive:
            assert json.loads(archive.read("manifest.json"))["format_version"] == 1
            assert archive.read("study/source/record.edf") == b"test recording"

        deleted = client.delete(f"/api/v1/sessions/{session.id}")
        assert deleted.status_code == 204
        assert client.get(f"/api/v1/exports/{export_id}").status_code == 404
        assert client.get(f"/api/v1/exports/{export_id}/download").status_code == 404

    assert not (storage.root / session.id).exists()


@pytest.mark.parametrize("payload", [
    {"format": "json", "mode": "workspace"},
    {"format": "sleepwalker", "mode": "results"},
    {"format": "sleepwalker", "mode": "workspace", "include": {"metadata": False}},
])
def test_export_rejects_invalid_format_mode_combinations(payload) -> None:
    session_service = SessionService(SessionStore())
    session = _ready_session(session_service)
    with TestClient(create_app(session_service=session_service)) as client:
        response = client.post(f"/api/v1/sessions/{session.id}/exports", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
