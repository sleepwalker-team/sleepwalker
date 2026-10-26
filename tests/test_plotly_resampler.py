import numpy as np
import pytest

pytest.importorskip("plotly_resampler", reason="Install sleepwalker/server/requirements-plotly.txt for Plotly integration tests")

from dash import Dash, dcc, html, no_update

from sleepwalker.server.services.plotly_service import HighFrequencySignal, PlotlyService
from sleepwalker.server.services.session_service import SessionService
from sleepwalker.server.stores.session_store import SessionState, SessionStore
from sleepwalker.server.validation.study import SignalInfo, StudyFormat


class ArrayReader:
    def __init__(self):
        self.calls = 0
        self.times = np.arange(100_000, dtype=float) / 1000
        self.values = np.sin(self.times)
        self.values[50_123] = 20.

    def read(self, _path, signals):
        self.calls += 1
        return [HighFrequencySignal(signal.id, self.times, self.values) for signal in signals]


def setup_plot():
    sessions = SessionService(SessionStore())
    session = sessions.create(["recording.edf"])
    session.state, session.format = SessionState.READY, StudyFormat.EDF
    session.signals = [SignalInfo("eeg", "EEG", "uV", 1000)]
    reader = ArrayReader()
    plots = PlotlyService(sessions, signal_reader=reader)
    initial = plots.create_figure(session.id, ["eeg"], max_points=100)
    return plots, session, reader, initial


def patch_values(patch):
    operations = patch.to_plotly_json()["operations"]
    return {tuple(op["location"]): op["params"]["value"] for op in operations if op["operation"] == "Assign"}


def test_initial_figure_is_small_but_zoom_uses_original_samples():
    plots, session, reader, initial = setup_plot()
    assert len(initial["data"][0]["x"]) <= 100
    assert initial["layout"]["yaxis"]["title"]["text"] == "uV"
    assert len(plots._views[(session.id, "main")].figure.hf_data[0]["y"]) == 100_000
    patch = plots.update_figure(session.id, {"xaxis.range[0]": 50.12, "xaxis.range[1]": 50.13})
    values = patch_values(patch)
    x, y = values[("data", 0, "x")], values[("data", 0, "y")]
    assert len(x) <= 100
    assert 50.123 in x
    assert 20. in y
    assert reader.calls == 1


def test_pan_reset_and_irrelevant_events_use_resampler_without_edf_reads():
    plots, session, reader, _ = setup_plot()
    patch = plots.update_figure(session.id, {"xaxis.range[0]": 10., "xaxis.range[1]": 11.})
    x = patch_values(patch)[("data", 0, "x")]
    assert len(x) <= 100
    assert x[0] >= 9.99 and x[-1] <= 11.01
    assert plots.update_figure(session.id, {"xaxis.autorange": True}) is no_update
    reset = plots.update_figure(session.id, {"xaxis.autorange": True, "xaxis.showspikes": False})
    full_x = patch_values(reset)[("data", 0, "x")]
    assert len(full_x) <= 100
    assert full_x[0] == 0 and full_x[-1] == 99.999
    assert plots.update_figure(session.id, {"autosize": True}) is no_update
    assert reader.calls == 1


def test_dash_callback_returns_patch_and_no_update_after_session_deletion():
    plots, session, _, initial = setup_plot()
    dash_app = Dash(__name__)
    dash_app.layout = html.Div([dcc.Store(id="selected-session", data=session.id), dcc.Graph(id="signals", figure=initial)])
    callback = plots.register_dash_callback(dash_app, graph_id="signals", session_store_id="selected-session")
    patch = callback({"xaxis.range[0]": 10., "xaxis.range[1]": 10.01}, session.id)
    assert len(patch_values(patch)[("data", 0, "x")]) <= 100
    assert any("signals.figure" in key for key in dash_app.callback_map)
    plots.session_service.delete(session.id)
    assert callback({"xaxis.range[0]": 10., "xaxis.range[1]": 11.}, session.id) is no_update
    assert callback({}, None) is no_update


def test_dash_http_transport_serializes_only_reduced_figure_and_patch():
    plots, session, _, initial = setup_plot()
    dash_app = Dash(__name__)
    dash_app.layout = html.Div([dcc.Store(id="selected-session", data=session.id), dcc.Graph(id="signals", figure=initial)])
    plots.register_dash_callback(dash_app, graph_id="signals", session_store_id="selected-session")
    client = dash_app.server.test_client()
    layout = client.get("/_dash-layout")
    assert layout.status_code == 200
    assert len(layout.data) < 50_000
    output = next(iter(dash_app.callback_map))
    response = client.post("/_dash-update-component", json={
        "output": output, "outputs": {"id": "signals", "property": "figure"},
        "inputs": [{"id": "signals", "property": "relayoutData", "value": {"xaxis.range[0]": 50.12, "xaxis.range[1]": 50.13}}],
        "state": [{"id": "selected-session", "property": "data", "value": session.id}],
        "changedPropIds": ["signals.relayoutData"],
    })
    assert response.status_code == 200
    assert len(response.data) < 20_000
    operations = response.get_json()["response"]["signals"]["figure"]["operations"]
    values = {tuple(op["location"]): op["params"]["value"] for op in operations if op["operation"] == "Assign"}
    assert len(values[("data", 0, "x")]) <= 100
    assert 20. in values[("data", 0, "y")]


def test_real_edf_plotting_keeps_original_resolution():
    from pathlib import Path
    sessions = SessionService(SessionStore())
    path = Path(__file__).parent / "data" / "signals_01.edf"
    session = sessions.create([path])
    session.state, session.format = SessionState.READY, StudyFormat.EDF
    session.signals = [SignalInfo("EEG", "EEG", "uV", 200)]
    plots = PlotlyService(sessions)
    initial = plots.create_figure(session.id, ["EEG"], max_points=100)
    assert len(initial["data"][0]["x"]) <= 100
    assert len(plots._views[(session.id, "main")].figure.hf_data[0]["y"]) == 720_000
    patch = plots.update_figure(session.id, {"xaxis.range[0]": 10., "xaxis.range[1]": 10.05})
    x = patch_values(patch)[("data", 0, "x")]
    assert len(x) <= 100
    assert np.diff(x).min() == pytest.approx(1 / 200)
