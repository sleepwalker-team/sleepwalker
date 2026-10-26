import builtins
import weakref
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from sleepwalker.server.services.plotly_service import (
    EdfPlotSignalReader, HighFrequencySignal, PlotFigureNotFoundError,
    PlotlyService, PlotlyUnavailableError, PlotSessionNotFoundError,
)
from sleepwalker.server.services.session_service import SessionService
from sleepwalker.server.services.signal_reader import SignalNotFoundError
from sleepwalker.server.stores.session_store import SessionState, SessionStore
from sleepwalker.server.validation.study import SignalInfo, StudyFormat


def make_session(service, source_path=Path("recording.edf")):
    session = service.create([source_path])
    session.state = SessionState.READY
    session.format = StudyFormat.EDF
    session.duration = 100.
    session.signals = [SignalInfo("eeg", "EEG", "uV", 100.)]
    return session


class FakeReader:
    def __init__(self):
        self.calls = []

    def read(self, path, signals):
        self.calls.append((path, [signal.id for signal in signals]))
        times = np.arange(10_000, dtype=float) / 100
        return [HighFrequencySignal(signal.id, times, np.sin(times)) for signal in signals]


class FakeFigure:
    def __init__(self, signals, max_points):
        self.max_points = max_points
        self.data = []
        self.updates = []

    def add_trace(self, trace, *, hf_x, hf_y, row, col):
        self.data.append((trace, hf_x, hf_y))

    def to_plotly_json(self):
        return {"data": [{"name": trace[0]["name"]} for trace in self.data], "layout": {}}

    def construct_update_data_patch(self, relayout):
        self.updates.append(relayout)
        return {"patch": relayout}


def make_service():
    sessions = SessionService(SessionStore())
    reader = FakeReader()
    return PlotlyService(sessions, signal_reader=reader, figure_factory=FakeFigure), sessions, reader


def test_plot_views_are_scoped_to_session_and_view_and_do_not_reread_on_zoom():
    plots, sessions, reader = make_service()
    first, second = make_session(sessions), make_session(sessions, Path("other.edf"))
    plots.create_figure(first.id, ["eeg"])
    plots.create_figure(second.id, ["eeg"])
    plots.create_figure(first.id, ["eeg"], view_id="second-graph")
    relayout = {"xaxis.range[0]": 10, "xaxis.range[1]": 11}
    assert plots.update_figure(first.id, relayout) == {"patch": relayout}
    assert len(reader.calls) == 3
    assert plots._views[(first.id, "main")].figure.updates == [relayout]
    assert plots._views[(second.id, "main")].figure.updates == []
    assert plots._views[(first.id, "second-graph")].figure.updates == []


def test_replacing_or_closing_figure_drops_old_state():
    plots, sessions, _ = make_service()
    session = make_session(sessions)
    plots.create_figure(session.id, ["eeg"])
    old_figure = weakref.ref(plots._views[(session.id, "main")].figure)
    plots.create_figure(session.id, ["eeg"])
    assert old_figure() is None
    current_figure = weakref.ref(plots._views[(session.id, "main")].figure)
    assert plots.delete_figure(session.id)
    assert current_figure() is None
    assert not plots.delete_figure(session.id)
    with pytest.raises(PlotFigureNotFoundError):
        plots.update_figure(session.id, {})


def test_deleting_session_releases_only_its_figures():
    plots, sessions, _ = make_service()
    first, second = make_session(sessions), make_session(sessions)
    plots.create_figure(first.id, ["eeg"])
    plots.create_figure(second.id, ["eeg"])
    assert plots.delete_session(first.id) == 1
    assert not plots.has_figure(first.id)
    assert plots.has_figure(second.id)
    plots.clear()
    assert not plots.has_figure(second.id)


def test_zoom_renews_session_and_expired_session_rejects_updates():
    now = datetime(2026, 10, 14, tzinfo=timezone.utc)
    sessions = SessionService(SessionStore(), clock=lambda: now)
    plots = PlotlyService(sessions, signal_reader=FakeReader(), figure_factory=FakeFigure)
    session = make_session(sessions)
    plots.create_figure(session.id, ["eeg"])
    now += timedelta(hours=1)
    plots.update_figure(session.id, {})
    assert session.expires_at == now + timedelta(hours=2)
    now += timedelta(hours=3)
    with pytest.raises(PlotSessionNotFoundError):
        plots.update_figure(session.id, {})
    assert not plots.has_figure(session.id)


def test_delete_during_signal_read_does_not_resurrect_plot():
    plots, sessions, reader = make_service()
    session = make_session(sessions)
    original = reader.read

    def read_and_delete(path, signals):
        data = original(path, signals)
        sessions.delete(session.id)
        plots.delete_session(session.id)
        return data

    reader.read = read_and_delete
    with pytest.raises(PlotSessionNotFoundError):
        plots.create_figure(session.id, ["eeg"])
    assert not plots.has_figure(session.id)


def test_delete_during_update_discards_patch():
    plots, sessions, _ = make_service()
    session = make_session(sessions)
    plots.create_figure(session.id, ["eeg"])

    def update_and_delete(_relayout):
        sessions.delete(session.id)
        plots.delete_session(session.id)
        return {"stale": True}

    plots._views[(session.id, "main")].figure.construct_update_data_patch = update_and_delete
    with pytest.raises(PlotSessionNotFoundError):
        plots.update_figure(session.id, {})


def test_closing_view_during_signal_read_does_not_recreate_it():
    plots, sessions, reader = make_service()
    session = make_session(sessions)
    original = reader.read

    def read_and_close(path, signals):
        data = original(path, signals)
        assert plots.delete_figure(session.id)
        return data

    reader.read = read_and_close
    with pytest.raises(PlotFigureNotFoundError):
        plots.create_figure(session.id, ["eeg"])
    assert not plots.has_figure(session.id)
    assert plots._pending == {}


def test_newer_selection_wins_if_previous_signal_read_finishes_later():
    plots, sessions, reader = make_service()
    session = make_session(sessions)
    original = reader.read

    def read_with_new_selection(path, signals):
        data = original(path, signals)
        if len(reader.calls) == 1:
            plots.create_figure(session.id, ["eeg"], max_points=300)
        return data

    reader.read = read_with_new_selection
    with pytest.raises(PlotFigureNotFoundError):
        plots.create_figure(session.id, ["eeg"], max_points=20)
    assert plots._views[(session.id, "main")].figure.max_points == 300
    assert plots._pending == {}


@pytest.mark.parametrize("signals,max_points,view_id", [
    ([], 20, "main"), (["eeg", "eeg"], 20, "main"), ("eeg", 20, "main"),
    (["eeg"], 2, "main"), (["eeg"], True, "main"), (["eeg"], 100001, "main"), (["eeg"], 20, ""),
])
def test_invalid_plot_parameters_are_rejected_before_loading(signals, max_points, view_id):
    plots, sessions, reader = make_service()
    session = make_session(sessions)
    with pytest.raises(ValueError):
        plots.create_figure(session.id, signals, max_points=max_points, view_id=view_id)
    assert reader.calls == []


def test_unknown_signal_and_unready_session_are_rejected():
    plots, sessions, reader = make_service()
    session = make_session(sessions)
    with pytest.raises(SignalNotFoundError):
        plots.create_figure(session.id, ["missing"])
    session.state = SessionState.PROCESSING
    with pytest.raises(ValueError):
        plots.create_figure(session.id, ["eeg"])
    assert reader.calls == []


def test_plot_dependencies_are_optional_and_missing_dependencies_fail_before_edf_read(monkeypatch, tmp_path):
    original = builtins.__import__

    def without_plotly(name, *args, **kwargs):
        if name.split(".")[0] in {"plotly", "plotly_resampler", "dash"}:
            raise ImportError("Optional plotting disabled for this test")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_plotly)
    from sleepwalker.server.app import create_app
    from sleepwalker.server.storage.temporary_storage import TemporaryStorage
    app = create_app(temporary_storage=TemporaryStorage(tmp_path / "storage"))
    assert isinstance(app.state.plotly_service, PlotlyService)
    session = make_session(app.state.session_service)
    reader = FakeReader()
    app.state.plotly_service.signal_reader = reader
    with pytest.raises(PlotlyUnavailableError, match="requirements-plotly"):
        app.state.plotly_service.create_figure(session.id, ["eeg"])
    assert reader.calls == []


def test_full_resolution_reader_keeps_all_samples_and_closes_file(monkeypatch):
    class Reader:
        closed = False
        read_sizes = []

        def getSignalLabels(self):
            return [" EEG ", "EOG"]

        def getNSamples(self):
            return [100_000, 100_000]

        def getSampleFrequency(self, _index):
            return 100.

        def readSignal(self, index, start, n):
            self.read_sizes.append(n)
            return np.arange(start, start + n, dtype=float) + index

        def close(self):
            self.closed = True

    reader = Reader()
    monkeypatch.setattr("sleepwalker.server.services.plotly_service.pyedflib.EdfReader", lambda _: reader)
    signals = [SignalInfo("eeg", "EEG", "uV", 100), SignalInfo("eog", "EOG", "uV", 100)]
    data = EdfPlotSignalReader().read(Path("test.edf"), signals)
    assert reader.closed
    assert max(reader.read_sizes) <= 65536
    assert len(data[0].values) == 100_000
    assert data[0].times is data[1].times
    assert data[0].times[-1] == 999.99
    assert data[0].values[-1] == 99999
    assert data[1].values[-1] == 100000


def test_full_resolution_reader_closes_file_on_missing_signal(monkeypatch):
    class Reader:
        closed = False

        def getSignalLabels(self):
            return []

        def getNSamples(self):
            return []

        def close(self):
            self.closed = True

    reader = Reader()
    monkeypatch.setattr("sleepwalker.server.services.plotly_service.pyedflib.EdfReader", lambda _: reader)
    with pytest.raises(SignalNotFoundError):
        EdfPlotSignalReader().read(Path("test.edf"), [SignalInfo("eeg", "EEG", "uV", 100)])
    assert reader.closed
