import json
from zipfile import ZipFile

from sleepwalker.server.app import create_app, create_local_app
from sleepwalker.server.services.package_inference_runner import PackageInferenceRunner
from sleepwalker.server.services.model_registry import ModelRegistry

from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from sleepwalker.server.services.session_service import SessionService
from sleepwalker.server.stores.session_store import SessionStore


def test_create_app_uses_empty_registry_by_default() -> None:
    app = create_app()

    assert app.state.model_registry.list() == []


def test_create_app_uses_provided_model_registry() -> None:
    registry = ModelRegistry()

    app = create_app(model_registry=registry)

    assert app.state.model_registry is registry


def test_create_local_app_connects_registry_store_and_runner(tmp_path) -> None:
    package_path = tmp_path / "sleep-model.swmodel"
    metadata = {
        "format": "sleepwalker-swmodel-v1",
        "model": {
            "id": "sleep-model",
            "name": "Sleep model",
            "task": "sleep_stage",
            "classes": ["wake", "n1"],
            "input_channels": ["eeg"],
            "capabilities": ["classification"],
            "output_resolution": 30.0,
        },
    }

    with ZipFile(package_path, "w") as package:
        package.writestr("meta.json", json.dumps(metadata))

    app = create_local_app(tmp_path, device="cpu", batch_size=16)

    assert app.state.model_registry.get("sleep-model") is not None
    assert isinstance(app.state.inference_runner, PackageInferenceRunner)
    assert app.state.inference_runner.prediction_store is app.state.prediction_store
    assert app.state.inference_runner.device == "cpu"
    assert app.state.inference_runner.batch_size == 16

def test_app_cleans_expired_sessions_on_startup() -> None:
    created_at = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    startup_at = created_at + timedelta(hours=3)
    times = iter([created_at, startup_at])

    session_service = SessionService(
        SessionStore(),
        ttl=timedelta(hours=2),
        clock=lambda: next(times),
    )
    session = session_service.create(source_files=[])

    app = create_app(session_service=session_service)

    with TestClient(app):
        assert session_service.store.get(session.id) is None