from fastapi import FastAPI

from .api import health, models, sessions
from .errors import register_exception_handlers


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sleepwalker API",
        version="0.1.0",
    )

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(sessions.router)

    return app


app = create_app()
