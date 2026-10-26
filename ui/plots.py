"""Small Dash adapter for the API's public Python PlotlyService interface."""

import copy
import math
import re
import time
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from threading import RLock

from dash import Dash, Input, Output, Patch, State, ctx, dcc, html, no_update

from sleepwalker.server.services.plotly_service import (
    PlotFigureNotFoundError, PlotlyUnavailableError, PlotSessionNotFoundError,
)


def time_range(data, duration):
    """Accept Plotly relayout shapes, including matched subplot axes."""
    if not isinstance(data, dict):
        return None
    for key, value in data.items():
        axis = key.split(".")[0]
        if not re.fullmatch(r"xaxis\d*", axis):
            continue
        if key.endswith(".autorange") and value:
            return [0.0, duration]
        pair = value if key.endswith(".range") else [data.get(axis + ".range[0]"), data.get(axis + ".range[1]")]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            start, end = (float(x) for x in pair)
        except (TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            width = min(end - start, duration)
            start = max(0.0, min(start, duration - width))
            return [start, start + width]
    return None


def style_figure(figure, signals, bounds, revision):
    figure = copy.deepcopy(figure)
    layout = figure.setdefault("layout", {})
    layout.update({
        "template": {}, "paper_bgcolor": "#fcfcfb", "plot_bgcolor": "#fcfcfb",
        "font": {"family": "Arial, sans-serif", "size": 11, "color": "#343c40"},
        "margin": {"l": 126, "r": 22, "t": 16, "b": 12},
        "height": max(230, len(signals) * 100), "showlegend": False,
        "dragmode": "pan", "hovermode": "closest", "uirevision": revision,
        "annotations": [],
    })
    for index, signal in enumerate(signals):
        axis = "" if index == 0 else str(index + 1)
        layout["xaxis" + axis].update({
            "range": bounds, "autorange": False, "title": None,
            "showticklabels": False, "showgrid": True, "gridcolor": "#e9eae8",
            "zeroline": False, "fixedrange": False,
        })
        yaxis = layout["yaxis" + axis]
        yaxis.update({"title": None, "gridcolor": "#eff0ed", "zeroline": False, "nticks": 3, "fixedrange": True})
        domain = yaxis.get("domain", [0, 1])
        layout["annotations"].append({
            "text": escape(signal["label"]), "x": -.02, "xref": "paper", "y": sum(domain) / 2,
            "yref": "paper", "showarrow": False, "xanchor": "right", "yanchor": "bottom",
            "font": {"size": 11, "color": "#262a2c"},
        })
        layout["annotations"].append({
            "text": escape(signal.get("unit", "")), "x": -.02, "xref": "paper", "y": sum(domain) / 2,
            "yref": "paper", "showarrow": False, "xanchor": "right", "yanchor": "top",
            "font": {"size": 10, "color": "#6e7476"},
        })
        figure["data"][index].update({"line": {"color": "#343c40", "width": 1}, "name": signal["label"]})
    return figure


def apply_assignments(figure, patch):
    """Apply the resampler's initial data patch to the reduced initial figure."""
    if patch is no_update:
        return
    for operation in patch.to_plotly_json()["operations"]:
        if operation["operation"] != "Assign":
            raise ValueError("Unsupported initial resampler patch operation.")
        target = figure
        for key in operation["location"][:-1]:
            target = target[key]
        target[operation["location"][-1]] = operation["params"]["value"]


@dataclass
class View:
    channels: tuple = ()
    initial: dict | None = None
    accessed: float = field(default_factory=time.monotonic)
    revision: int = -1
    lock: RLock = field(default_factory=RLock)


class PlotController:
    def __init__(self, service, *, idle_seconds=180, clock=time.monotonic):
        self.service = service
        self.idle_seconds = idle_seconds
        self.clock = clock
        self.views = {}
        self.lock = RLock()

    def render(self, request, relayout=None):
        if not isinstance(request, dict):
            return no_update, no_update
        session_id, view_id = request.get("session"), request.get("view")
        if not isinstance(session_id, str) or not isinstance(view_id, str) or not view_id or len(view_id) > 120:
            return no_update, {"error": "Ungültige Plotansicht."}
        key = (session_id, view_id)
        command = request.get("action", "view")
        if command == "close":
            with self.lock:
                view = self.views.pop(key, None)
            if view:
                with view.lock:
                    self.service.delete_figure(session_id, view_id=view_id)
            return {"data": [], "layout": {}}, {"closed": True, "session": session_id, "view": view_id}
        with self.lock:
            view = self.views.setdefault(key, View(accessed=self.clock()))
            view.lock.acquire()
        try:
            revision = request.get("revision", 0)
            if not isinstance(revision, int) or revision < view.revision:
                return no_update, no_update
            view.revision = revision
            view.accessed = self.clock()
            session = self.service.session_service.get(session_id)
            if session is None:
                self.service.delete_figure(session_id, view_id=view_id)
                return no_update, {"error": "Session nicht mehr verfügbar. EDF oder Workspace erneut öffnen.", "expired": True, "session": session_id, "view": view_id}
            if command == "heartbeat" and self.service.has_figure(session_id, view_id=view_id):
                return no_update, no_update
            channels = request.get("channels", [])
            available = {signal.id: {"label": signal.label, "unit": signal.unit} for signal in session.signals or []}
            if not isinstance(channels, list) or not channels or any(not isinstance(c, str) for c in channels) or len(channels) != len(set(channels)) or any(channel not in available for channel in channels):
                return no_update, {"error": "Mindestens einen gültigen Kanal auswählen.", "session": session_id, "view": view_id}
            duration = float(session.duration or 0)
            bounds = time_range(relayout, duration) if relayout else time_range({"xaxis.range": request.get("range")}, duration)
            if bounds is None:
                return no_update, no_update
            # Matched axes synchronize the browser viewport, but the resampler
            # filters data by each trace's own subplot axis. Request every axis
            # so lower channels are not left at the whole-recording resolution.
            viewport = {}
            for index in range(len(channels)):
                axis = "xaxis" + (str(index + 1) if index else "")
                viewport[axis + ".range[0]"] = bounds[0]
                viewport[axis + ".range[1]"] = bounds[1]
            try:
                created = view.channels != tuple(channels) or not self.service.has_figure(session_id, view_id=view_id)
                if created:
                    initial = self.service.create_figure(session_id, channels, view_id=view_id, max_points=2000)
                    patch = self.service.update_figure(session_id, viewport, view_id=view_id)
                    apply_assignments(initial, patch)
                    view.channels = tuple(channels)
                    result = style_figure(initial, [available[c] for c in channels], bounds, f"{session_id}:{view_id}:{','.join(channels)}")
                    view.initial = result
                else:
                    result = self.service.update_figure(session_id, viewport, view_id=view_id)
                    if request.get("initialize") and view.initial is not None:
                        initial = copy.deepcopy(view.initial)
                        apply_assignments(initial, result)
                        for i in range(len(channels)):
                            axis = "xaxis" + (str(i + 1) if i else "")
                            initial["layout"][axis]["range"] = bounds
                        view.initial = initial
                        return initial, {"session": session_id, "view": view_id, "range": bounds, "revision": revision, "height": max(230, len(channels) * 100)}
                    if result is no_update:
                        result = Patch()
                    for i in range(len(channels)):
                        axis = "xaxis" + (str(i + 1) if i else "")
                        result["layout"][axis]["range"] = bounds
                        result["layout"][axis]["autorange"] = False
                return result, {"session": session_id, "view": view_id, "range": bounds, "revision": revision, "height": max(230, len(channels) * 100)}
            except (PlotSessionNotFoundError, PlotFigureNotFoundError):
                return no_update, no_update
            except (ValueError, PlotlyUnavailableError) as exc:
                return no_update, {"error": str(exc), "session": session_id, "view": view_id}
        finally:
            view.lock.release()

    def cleanup(self):
        now = self.clock()
        with self.lock:
            stale = [(key, view) for key, view in self.views.items() if now - view.accessed > self.idle_seconds]
            for key, view in stale:
                if not view.lock.acquire(blocking=False):
                    continue
                try:
                    if now - view.accessed > self.idle_seconds:
                        self.service.delete_figure(key[0], view_id=key[1])
                        self.views.pop(key, None)
                finally:
                    view.lock.release()

    def clear(self):
        with self.lock:
            for session_id, view_id in list(self.views):
                self.service.delete_figure(session_id, view_id=view_id)
            self.views.clear()


def create_dash_app(controller):
    app = Dash(__name__, requests_pathname_prefix="/ui/plot/", routes_pathname_prefix="/",
               assets_folder=str(Path(__file__).parent / "plot_assets"), title="Sleepwalker · Signale",
               update_title=None, external_scripts=[], external_stylesheets=[])
    app.layout = html.Div([
        dcc.Store(id="plot-request"), dcc.Store(id="plot-status"), html.Div(id="bridge-output", hidden=True),
        dcc.Graph(id="signals", figure={"data": [], "layout": {"paper_bgcolor": "#fcfcfb", "plot_bgcolor": "#fcfcfb"}},
                  config={"displaylogo": False, "displayModeBar": False, "scrollZoom": True, "responsive": True},
                  # The iframe gets its height from the channel count. A percentage
                  # here collapses to zero when Graph's responsive React update
                  # removes layout.height and its auto-height parents collapse.
                  style={"width": "100%", "height": "100vh"}),
    ])

    @app.callback(Output("signals", "figure"), Output("plot-status", "data"),
                  Input("plot-request", "data"), prevent_initial_call=True)
    def render(request):
        return controller.render(request)

    app.clientside_callback(
        """function(status, click, relayout) {
            const prop = window.dash_clientside.callback_context.triggered?.[0]?.prop_id;
            if (window.sleepwalkerBridge) window.sleepwalkerBridge.report(status, click, prop === 'signals.relayoutData' ? relayout : null);
            return window.dash_clientside.no_update;
        }""",
        Output("bridge-output", "children"), Input("plot-status", "data"), Input("signals", "clickData"), Input("signals", "relayoutData"),
    )
    return app
