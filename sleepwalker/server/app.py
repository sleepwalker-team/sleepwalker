import os
from pathlib import Path

from fastapi import FastAPI

from .api import health, models, sessions
from .errors import register_exception_handlers
from .services.model_registry import ModelRegistry, load_model_registry

MODEL_DIRECTORY_ENV = "SLEEPWALKER_MODEL_DIR"


def load_model_registry_from_environment() -> ModelRegistry:
    configured_directory = os.getenv(MODEL_DIRECTORY_ENV)

    if (
        configured_directory is None
        or not configured_directory.strip()
    ):
        return ModelRegistry()

    return load_model_registry(
        Path(configured_directory)
    )

def create_app(
    model_registry: ModelRegistry | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Sleepwalker API",
        version="0.1.0",
    )

    app.state.model_registry = (
        model_registry
        if model_registry is not None
        else load_model_registry_from_environment()
    )

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(sessions.router)

    return app


app = create_app()
