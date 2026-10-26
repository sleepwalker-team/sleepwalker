"""ASGI composition: UI routes and the original API share one lifespan."""

import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .plots import PlotController, create_dash_app

HERE = Path(__file__).parent


class RevalidatingStaticFiles(StaticFiles):
    """HTML and its modules must be from the same UI version after an update."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_ui_app(api):
    """Wrap an existing API app; never add to or replace its routes or schema."""
    plots = PlotController(api.state.plotly_service)
    dash_app = create_dash_app(plots)

    @asynccontextmanager
    async def lifespan(app):
        async with api.router.lifespan_context(api):
            async def reap_views():
                while True:
                    await asyncio.sleep(30)
                    await asyncio.to_thread(plots.cleanup)
            task = asyncio.create_task(reap_views())
            try:
                yield
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                plots.clear()

    async def index(request):
        response = FileResponse(HERE / "assets" / "index.html")
        response.headers["Cache-Control"] = "no-store"
        return response

    async def config(request):
        return JSONResponse({"ui": "sleepwalker", "version": 1, "plot": "/ui/plot/"})

    app = Starlette(routes=[
        Route("/", lambda request: RedirectResponse("/ui/")),
        Route("/ui", lambda request: RedirectResponse("/ui/")),
        Route("/ui/", index),
        Route("/ui/config", config),
        Mount("/ui/assets", app=RevalidatingStaticFiles(directory=HERE / "assets")),
        Mount("/ui/plot", app=WSGIMiddleware(dash_app.server)),
        Mount("/", app=api),
    ], lifespan=lifespan)
    app.state.api = api
    app.state.plots = plots
    app.state.dash = dash_app
    return app
