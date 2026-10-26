import asyncio
from threading import Event
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app, run_cleanup_in_thread
from sleepwalker.server.storage.temporary_storage import TemporaryStorage


def test_startup_removes_orphan_sessions_but_keeps_known_and_unrelated_files(tmp_path) -> None:
    storage = TemporaryStorage(tmp_path / "uploads")
    app = create_app(temporary_storage=storage)
    known = app.state.session_service.create([])
    known_directory = storage.root / known.id
    known_directory.mkdir()
    orphan = storage.write_json_export(f"sess_{uuid4()}", "export-test", {"test": True})
    unrelated = storage.root / "notes"
    unrelated.mkdir()
    note = unrelated / "note.txt"
    note.write_text("keep", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = storage.root / f"sess_{uuid4()}"
    link.symlink_to(outside, target_is_directory=True)

    with TestClient(app):
        assert not orphan.exists()
        assert known_directory.exists()
        assert note.read_text(encoding="utf-8") == "keep"
        assert link.is_symlink()
        assert outside.exists()


def test_restart_reclaims_previous_server_temporary_session(tmp_path) -> None:
    root = tmp_path / "uploads"
    first = create_app(temporary_storage=TemporaryStorage(root))
    with TestClient(first):
        session = first.state.session_service.create([])
        path = first.state.temporary_storage.write_json_export(session.id, "export-test", {"test": True})
    assert path.exists()

    second = create_app(temporary_storage=TemporaryStorage(root))
    with TestClient(second):
        assert second.state.session_service.store.list() == []
        assert not path.exists()


def test_second_server_cannot_claim_live_storage_and_lock_is_released(tmp_path) -> None:
    storage = TemporaryStorage(tmp_path / "uploads")
    first = create_app(temporary_storage=storage)
    second = create_app(temporary_storage=TemporaryStorage(storage.root))
    with TestClient(first):
        session = first.state.session_service.create([])
        path = storage.write_json_export(session.id, "export-test", {"test": True})
        with pytest.raises(RuntimeError, match="already used"):
            with TestClient(second):
                pass
        assert path.exists()

    with TestClient(second):
        assert not path.exists()


def test_orphan_cleanup_requires_exclusive_ownership(tmp_path) -> None:
    storage = TemporaryStorage(tmp_path / "uploads")
    path = storage.write_json_export(f"sess_{uuid4()}", "export-test", {"test": True})
    with pytest.raises(RuntimeError, match="ownership"):
        storage.cleanup_orphan_sessions(set())
    assert path.exists()


def test_shutdown_waits_for_running_cleanup_before_releasing_ownership() -> None:
    async def scenario():
        started = Event()
        release = Event()
        completed = Event()

        def cleanup():
            started.set()
            assert release.wait(timeout=3.0)
            completed.set()

        task = asyncio.create_task(run_cleanup_in_thread(cleanup))
        assert await asyncio.to_thread(started.wait, 1.0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed.is_set()

    asyncio.run(scenario())
