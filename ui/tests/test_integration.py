from pathlib import Path
import re

import numpy as np
import pytest
from dash import no_update
from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.storage.temporary_storage import TemporaryStorage
from sleepwalker.server.services.plotly_service import HighFrequencySignal
from sleepwalker.server.stores.session_store import SessionState
from sleepwalker.server.validation.study import SignalInfo, StudyFormat

from ui.app import create_ui_app
from ui.plots import PlotController, apply_assignments, time_range

EDF = Path(__file__).resolve().parents[2] / "tests" / "data" / "signals_01.edf"


class Reader:
    def __init__(self):
        self.calls = 0
        self.x = np.arange(100_000, dtype=float) / 1000
        self.y = np.sin(self.x)
        self.y[50_123] = 20

    def read(self, path, signals):
        self.calls += 1
        return [HighFrequencySignal(signal.id, self.x, self.y) for signal in signals]


@pytest.fixture
def api(tmp_path):
    return create_app(temporary_storage=TemporaryStorage(tmp_path / "server"))


def prepared(api):
    session = api.state.session_service.create([Path("synthetic.edf")])
    session.state, session.format, session.duration = SessionState.READY, StudyFormat.EDF, 100
    session.signals = [SignalInfo("EEG", "EEG", "uV", 1000), SignalInfo("EOG", "EOG", "uV", 1000)]
    reader = Reader()
    api.state.plotly_service.signal_reader = reader
    return session, reader


def command(session, view="tab-1", **changes):
    return {"session": session.id, "view": view, "channels": ["EEG"], "range": [50.12, 50.13], "revision": 1, **changes}


def test_composition_preserves_api_contract_and_serves_local_assets(api):
    schema, routes = api.openapi(), tuple(api.routes)
    ui = create_ui_app(api)
    assert api.openapi() == schema
    assert tuple(api.routes) == routes
    assert ui.state.plots.service is api.state.plotly_service
    with TestClient(ui) as client:
        assert client.get("/").url.path == "/ui/"
        assert 'lang="de"' in client.get("/ui/").text
        assert client.get("/ui/config").json()["ui"] == "sleepwalker"
        assert client.get("/ui/assets/app.js").status_code == 200
        for asset in ["app.js", "timeline.js", "corrections.js", "styles.css"]:
            response = client.get(f"/ui/assets/{asset}?v=test-version")
            assert response.headers["Cache-Control"] == "no-cache"
            assert client.get(f"/ui/assets/{asset}?v=test-version", headers={"If-None-Match":response.headers["ETag"]}).status_code == 304
        assert client.get("/ui/plot/").status_code == 200
        assert client.get("/ui/plot/_dash-layout").status_code == 200
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/openapi.json").json() == schema
        assert not any(path.startswith("/ui") for path in schema["paths"])


def test_browser_modules_and_html_share_a_version_and_all_required_elements():
    assets = Path(__file__).resolve().parents[1] / "assets"
    html = (assets / "index.html").read_text()
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    source = (assets / "app.js").read_text()
    required = set(re.findall(r'\$\("([^"]+)"\)', source))
    required.update(re.findall(r'(?:show|text|bind)\("([^"]+)"', source))
    assert not required - ids
    version = re.search(r'/ui/assets/app.js\?v=([^"&]+)', html).group(1)
    assert f"/ui/assets/styles.css?v={version}" in html
    for name in ["app.js", "timeline.js"]:
        imports = re.findall(r'from "(\./[^"]+)"', (assets / name).read_text())
        assert imports
        assert all(url.endswith(f"?v={version}") for url in imports)


def test_initial_visible_range_uses_original_samples_without_second_read(api):
    session, reader = prepared(api)
    controller = PlotController(api.state.plotly_service)
    initial, status = controller.render(command(session))
    assert initial["layout"]["xaxis"]["range"] == [50.12, 50.13]
    assert initial["layout"]["yaxis"]["fixedrange"] is True
    assert initial["layout"]["xaxis"]["fixedrange"] is False
    assert 20 in initial["data"][0]["y"]
    assert len(initial["data"][0]["x"]) <= 2000
    patch, status = controller.render(command(session, revision=2), {"xaxis.range[0]": 10, "xaxis.range[1]": 10.2})
    assert patch is not no_update
    assert status["range"] == [10, 10.2]
    assert reader.calls == 1


def test_navigation_before_browser_initialization_returns_complete_reduced_figure(api):
    session, reader = prepared(api)
    ui = PlotController(api.state.plotly_service)
    ui.render(command(session, initialize=True))
    initial, status = ui.render(command(session, initialize=True, revision=2, range=[10., 10.2]))
    assert isinstance(initial, dict)
    assert initial["data"]
    assert initial["layout"]["xaxis"]["range"] == [10., 10.2]
    assert len(initial["data"][0]["x"]) <= 2000
    assert reader.calls == 1


def test_zoom_resamples_every_subplot_at_the_visible_resolution(api):
    session, reader = prepared(api)
    ui = PlotController(api.state.plotly_service)
    initial, _ = ui.render(command(session, channels=["EEG", "EOG"]))
    for trace in initial["data"]:
        assert 20 in trace["y"]
        assert len(trace["x"]) <= 12
    patch, _ = ui.render(command(session, channels=["EEG", "EOG"], revision=2, range=[20., 20.1]))
    apply_assignments(initial, patch)
    for trace in initial["data"]:
        assert 99 <= len(trace["x"]) <= 102
        assert trace["x"][0] >= 20.
        assert trace["x"][-1] <= 20.1
    assert reader.calls == 1


def test_real_api_edf_upload_and_resampler_share_session(api):
    ui = create_ui_app(api)
    with TestClient(ui) as client:
        with EDF.open("rb") as source:
            response = client.post("/api/v1/sessions", files={"files": ("night.edf", source, "application/octet-stream")})
        assert response.status_code == 201
        session = response.json()
        request = {"session": session["id"], "view": "edf-tab", "channels": [session["signals"][0]["id"]], "range": [10, 10.05], "revision": 1}
        initial, _ = ui.state.plots.render(request)
        assert len(initial["data"][0]["x"]) <= 2000
        assert np.diff(initial["data"][0]["x"]).min() <= .01
        assert client.delete(f'/api/v1/sessions/{session["id"]}').status_code == 204
        assert not api.state.plotly_service.has_figure(session["id"], view_id="edf-tab")


def test_tabs_and_channel_replacement_do_not_share_or_delete_views(api):
    session, reader = prepared(api)
    ui = PlotController(api.state.plotly_service)
    ui.render(command(session, "one"))
    ui.render(command(session, "two", channels=["EOG"]))
    assert reader.calls == 2
    ui.render(command(session, "one", action="close", revision=2))
    assert not api.state.plotly_service.has_figure(session.id, view_id="one")
    assert api.state.plotly_service.has_figure(session.id, view_id="two")
    assert api.state.session_service.get(session.id) is session
    initial, _ = ui.render(command(session, "two", channels=["EEG", "EOG"], revision=2))
    assert len(initial["data"]) == 2
    assert reader.calls == 3


def test_stale_commands_cannot_replace_current_channel_selection(api):
    session, reader = prepared(api)
    ui = PlotController(api.state.plotly_service)
    ui.render(command(session, channels=["EOG"], revision=5))
    assert ui.render(command(session, channels=["EEG"], revision=4)) == (no_update, no_update)
    assert reader.calls == 1


def test_idle_cleanup_releases_arrays_but_keeps_session(api):
    session, _ = prepared(api)
    now = [0]
    ui = PlotController(api.state.plotly_service, idle_seconds=30, clock=lambda: now[0])
    ui.render(command(session))
    now[0] = 31
    ui.cleanup()
    assert not api.state.plotly_service.has_figure(session.id, view_id="tab-1")
    assert api.state.session_service.get(session.id) is session
    ui.render(command(session, revision=2))
    assert api.state.plotly_service.has_figure(session.id, view_id="tab-1")


@pytest.mark.parametrize("data,expected", [
    ({"xaxis3.range[0]": 5, "xaxis3.range[1]": 7}, [5, 7]),
    ({"xaxis.range": [-2, 8]}, [0, 10]),
    ({"xaxis.autorange": True}, [0, 100]),
    ({"xaxis.range": [95, 105]}, [90, 100]),
    ({"shapes[0].x0": 10}, None),
    ({"xaxis.range": [float("nan"), 1]}, None),
    ({"xaxis.range": [8, 4]}, None),
])
def test_time_range_normalizes_subplots_and_rejects_non_navigation(data, expected):
    assert time_range(data, 100) == expected
