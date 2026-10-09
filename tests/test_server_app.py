from sleepwalker.server.app import create_app
from sleepwalker.server.services.model_registry import ModelRegistry


def test_create_app_uses_empty_registry_by_default() -> None:
    app = create_app()

    assert app.state.model_registry.list() == []


def test_create_app_uses_provided_model_registry() -> None:
    registry = ModelRegistry()

    app = create_app(model_registry=registry)

    assert app.state.model_registry is registry