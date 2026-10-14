import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.services.job_service import JobService
from sleepwalker.server.storage.temporary_storage import TemporaryStorage
from sleepwalker.server.stores.job_store import JobError, JobPhase, JobState, JobStore


def _events(text):
    events = []
    for block in text.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


@pytest.mark.parametrize("state", [JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED])
def test_terminal_job_sends_one_event_with_public_job_schema_and_closes(state) -> None:
    service = JobService(JobStore())
    job = service.create("sess-test", "inference", {"secret": "private parameters"})
    if state == JobState.CANCELLED:
        service.request_cancel(job.id)
        service.mark_cancelled(job.id)
    else:
        service.mark_running(job.id, JobPhase.INFERENCE, "Running")
        if state == JobState.SUCCEEDED:
            service.mark_succeeded(job.id)
        else:
            service.mark_failed(job.id, JobError("MODEL_INCOMPATIBLE", "Incompatible model"))

    with TestClient(create_app(job_service=service)) as client:
        response = client.get(f"/api/v1/jobs/{job.id}/events")
        status = client.get(f"/api/v1/jobs/{job.id}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert _events(response.text) == [("job_update", status.json())]
    assert "private parameters" not in response.text
    assert service._subscriptions == {}


def test_events_for_unknown_job_return_normal_json_error() -> None:
    service = JobService(JobStore())
    with TestClient(create_app(job_service=service)) as client:
        response = client.get("/api/v1/jobs/job-missing/events")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "JOB_NOT_FOUND"
    assert service._subscriptions == {}


async def _request_stream(app, job_id, on_event):
    received = asyncio.Queue()
    received.put_nowait({"type": "http.request", "body": b"", "more_body": False})
    messages = []

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            events = _events(message["body"].decode())
            for event in events:
                on_event(event, received)

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "GET", "scheme": "http",
        "path": f"/api/v1/jobs/{job_id}/events", "root_path": "", "query_string": b"",
        "headers": [(b"host", b"testserver")], "server": ("testserver", 80), "client": ("client", 1),
    }
    await asyncio.wait_for(app(scope, received.get, send), timeout=2.0)
    text = b"".join(message.get("body", b"") for message in messages).decode()
    return _events(text)


def test_open_stream_delivers_running_progress_and_completion() -> None:
    async def scenario():
        service = JobService(JobStore())
        job = service.create("sess-test", "inference")

        def update(event, received):
            state = event[1]["state"]
            if state == "queued":
                service.mark_running(job.id, JobPhase.INFERENCE, "Running")
            elif state == "running" and event[1]["progress"] is None:
                service.update_progress(job.id, phase=JobPhase.INFERENCE, progress=0.5, message="Halfway")
                service.mark_succeeded(job.id)

        events = await _request_stream(create_app(job_service=service), job.id, update)
        assert [data["state"] for _, data in events] == ["queued", "running", "running", "succeeded"]
        assert events[2][1]["progress"] == 0.5
        assert service._subscriptions == {}

    asyncio.run(scenario())


def test_disconnected_client_releases_subscription_without_cancelling_job() -> None:
    async def scenario():
        service = JobService(JobStore())
        job = service.create("sess-test", "inference")

        def disconnect(event, received):
            received.put_nowait({"type": "http.disconnect"})

        events = await _request_stream(create_app(job_service=service), job.id, disconnect)
        assert len(events) == 1
        assert job.state == JobState.QUEUED
        assert service._subscriptions == {}

    asyncio.run(scenario())


def test_deleted_job_sends_deletion_event_and_closes_stream() -> None:
    async def scenario():
        service = JobService(JobStore())
        job = service.create("sess-test", "inference")

        def delete(event, received):
            if event[0] == "job_update":
                service.delete_session(job.session_id)

        events = await _request_stream(create_app(job_service=service), job.id, delete)
        assert [name for name, data in events] == ["job_update", "job_deleted"]
        assert events[-1][1]["error"]["code"] == "JOB_NOT_FOUND"
        assert service._subscriptions == {}

    asyncio.run(scenario())


def test_session_deletion_cancels_job_and_closes_stream(tmp_path) -> None:
    async def scenario():
        app = create_app(temporary_storage=TemporaryStorage(tmp_path / "uploads"))
        session = app.state.session_service.create([])
        job = app.state.job_service.create(session.id, "inference")

        def delete_and_cancel(event, received):
            state = event[1]["state"]
            if state == "queued":
                assert app.state.session_cleanup_service.request_delete(session.id) is True
            elif state == "cancelling":
                app.state.job_service.mark_cancelled(job.id)
                app.state.session_cleanup_service.finalize_pending_deletion(session.id)

        events = await _request_stream(app, job.id, delete_and_cancel)
        assert [data["state"] for _, data in events] == ["queued", "cancelling", "cancelled"]
        assert app.state.job_service.get(job.id) is None
        assert app.state.session_service.store.get(session.id) is None
        assert app.state.job_service._subscriptions == {}

    asyncio.run(scenario())
