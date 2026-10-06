from fastapi import FastAPI

from .api import health, sessions


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sleepwalker API",
        version="0.1.0",
    )

    app.include_router(health.router)
    app.include_router(sessions.router)

    return app


app = create_app()