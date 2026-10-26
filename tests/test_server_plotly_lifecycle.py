from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.services.plotly_service import HighFrequencySignal, PlotFigureNotFoundError, PlotSessionNotFoundError
from sleepwalker.server.stores.session_store import SessionState
from sleepwalker.server.validation.study import SignalInfo, StudyFormat


class FakeFigure:
    def __init__(self, _signals, _max_points):
        pass

    def add_trace(self, *_args, **_kwargs):
        pass

    def to_plotly_json(self):
        return {"data": [], "layout": {}}

    def construct_update_data_patch(self, _relayout):
        return {"update": True}


class FakeReader:
    def read(self, _path, signals):
        return [HighFrequencySignal(signal.id, np.arange(10.), np.arange(10.)) for signal in signals]


def prepare_plot(app):
    session = app.state.session_service.create([Path("recording.edf")])
    session.state, session.format = SessionState.READY, StudyFormat.EDF
    session.signals = [SignalInfo("eeg", "EEG", "uV", 100)]
    plots = app.state.plotly_service
    plots.figure_factory, plots.signal_reader = FakeFigure, FakeReader()
    plots.create_figure(session.id, ["eeg"])
    return session, plots


@pytest.mark.parametrize("active_job", [False, True])
def test_api_session_delete_immediately_removes_plot_even_with_active_job(active_job):
    app = create_app()
    with TestClient(app) as client:
        session, plots = prepare_plot(app)
        if active_job:
            app.state.job_service.create(session.id, "inference")
        response = client.delete(f"/api/v1/sessions/{session.id}")
        assert response.status_code == 204
        assert not plots.has_figure(session.id)
        with pytest.raises(PlotSessionNotFoundError):
            plots.update_figure(session.id, {})


def test_expiration_cleanup_removes_plot_state():
    app = create_app()
    session, plots = prepare_plot(app)
    session.expires_at = app.state.session_service.clock() - timedelta(seconds=1)
    assert app.state.session_cleanup_service.cleanup_expired() == [session.id]
    assert not plots.has_figure(session.id)


def test_app_shutdown_clears_plot_state():
    app = create_app()
    with TestClient(app):
        session, plots = prepare_plot(app)
        assert plots.has_figure(session.id)
    assert not plots.has_figure(session.id)


def test_plot_adapter_does_not_add_or_replace_rest_routes():
    app = create_app()
    paths = set(app.openapi()["paths"])
    assert "/api/v1/sessions/{session_id}/signals" in paths
    assert "/api/v1/sessions/{session_id}/signals/{signal_id}" in paths
    assert not any("plotly" in path or "figure" in path for path in paths)


def test_closing_view_does_not_delete_the_session():
    app = create_app()
    session, plots = prepare_plot(app)
    assert plots.delete_figure(session.id)
    assert app.state.session_service.store.get(session.id) is session
    with pytest.raises(PlotFigureNotFoundError):
        plots.update_figure(session.id, {})
