from pathlib import Path

from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.services.session_service import SessionService
from sleepwalker.server.services.signal_reader import SignalSamples
from sleepwalker.server.stores.session_store import SessionState, SessionStore
from sleepwalker.server.validation.study import SignalInfo


class FakeSignalReader:
    def read(
        self, _source_path: Path, _signal_id: str, start: float, end: float, *, max_points: int = 2000,
    ) -> SignalSamples:
        points = [(float(index), float(index - 5)) for index in range(10) if start <= index < end]
        return SignalSamples(
            times=[time for time, _ in points],
            values=[value for _, value in points],
        )


def create_signal_app():
    session_service = SessionService(SessionStore())
    session = session_service.create(source_files=[Path("/recordings/test.edf")])
    session.state = SessionState.READY
    session.duration = 10.0
    session.signals = [SignalInfo(id="EEG", label="EEG", unit="uV", sample_rate=1.0)]
    return create_app(session_service=session_service, signal_reader=FakeSignalReader()), session


def test_list_signals_returns_recording_bounds() -> None:
    app, session = create_signal_app()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/signals")

    assert response.status_code == 200
    assert response.json() == {
        "signals": [
            {
                "id": "EEG",
                "label": "EEG",
                "unit": "uV",
                "sample_rate": 1.0,
                "start": 0.0,
                "end": 10.0,
            }
        ]
    }


def test_get_signal_data_downsamples_requested_range() -> None:
    app, session = create_signal_app()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/signals/EEG?start=0&end=10&max_points=4")

    assert response.status_code == 200
    body = response.json()
    assert body["signal"] == "EEG"
    assert body["downsampled"] is True
    assert body["start"] == 0.0
    assert body["end"] == 10.0
    assert len(body["points"]) == 4
    assert body["points"][0] == [0.0, -5.0]
    assert body["points"][-1] == [9.0, 4.0]


def test_get_signal_data_rejects_invalid_max_points() -> None:
    app, session = create_signal_app()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/signals/EEG?max_points=1")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_MAX_POINTS"


def test_get_signal_data_rejects_unknown_signal() -> None:
    app, session = create_signal_app()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/signals/unknown")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SIGNAL_NOT_FOUND"


def test_real_edf_signal_query_returns_bounded_points_and_downsampled_flag() -> None:
    path = Path(__file__).parent / "data" / "signals_01.edf"
    sessions = SessionService(SessionStore())
    session = sessions.create([path])
    session.state = SessionState.READY
    session.duration = 3600.0
    session.signals = [SignalInfo(id="EEG", label="EEG", unit="uV", sample_rate=200.0)]
    with TestClient(create_app(session_service=sessions)) as client:
        response = client.get(f"/api/v1/sessions/{session.id}/signals/EEG?start=0&end=3600&max_points=23")
    assert response.status_code == 200
    assert len(response.json()["points"]) == 23
    assert response.json()["downsampled"] is True
    assert response.json()["points"][0][0] == 0.0
    assert response.json()["points"][-1][0] == 3599.995
