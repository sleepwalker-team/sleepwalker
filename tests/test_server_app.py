from pathlib import Path
import sys

from sleepwalker.server.app import create_app, create_local_app
from sleepwalker.server.inference.runner import PackageInferenceRunner
from sleepwalker.server.inference.catalog import ModelCatalog

from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from sleepwalker.server.core.sessions import SessionService


def test_create_app_uses_empty_registry_by_default() -> None:
    app = create_app()

    assert app.state.model_catalog.list() == []


def test_create_app_uses_provided_model_catalog() -> None:
    registry = ModelCatalog()

    app = create_app(model_catalog=registry)

    assert app.state.model_catalog is registry


def test_create_local_app_connects_registry_store_and_runner(tmp_path, make_model_package) -> None:
    make_model_package("sleep-model")
    app = create_local_app(tmp_path, inference_python=Path(sys.executable), device="cpu", batch_size=16)

    assert app.state.model_catalog.get("sleep-model") is not None
    assert isinstance(app.state.inference_runner, PackageInferenceRunner)
    assert app.state.inference_runner.prediction_store is app.state.prediction_store
    assert app.state.inference_runner.device == "cpu"
    assert app.state.inference_runner.batch_size == 16
    assert app.state.inference_runner.prediction_client.python_executable == Path(sys.executable).absolute()

def test_app_cleans_expired_sessions_on_startup() -> None:
    created_at = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    startup_at = created_at + timedelta(hours=3)
    times = iter([created_at, startup_at])

    session_service = SessionService(ttl=timedelta(hours=2),
        clock=lambda: next(times),
    )
    session = session_service.create(source_files=[])

    app = create_app(session_service=session_service)

    with TestClient(app):
        assert session_service.peek(session.id) is None
