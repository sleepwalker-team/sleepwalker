import numpy as np
import pytest

pytest.importorskip("plotly_resampler", reason="Install sleepwalker/server/requirements-plotly.txt for Plotly integration tests")

from dash import Dash, dcc, html, no_update

from sleepwalker.server.recording.plots import HighFrequencySignal, PlotlyService
from sleepwalker.server.core.sessions import SessionService
from sleepwalker.server.core.sessions import SessionState
from sleepwalker.server.recording.study import SignalInfo, StudyFormat


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
    sessions = SessionService()
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


def test_real_edf_plotting_keeps_original_resolution():
    from pathlib import Path
    sessions = SessionService()
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
