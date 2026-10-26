"""Optional, session-scoped Plotly transport, independent of REST signals."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

import numpy as np
import pyedflib

from .session_service import SessionService
from .signal_reader import SignalNotFoundError, _signal_chunks
from ..stores.session_store import Session, SessionState
from ..validation.study import SignalInfo, StudyFormat


class PlotlyUnavailableError(RuntimeError):
    pass


class PlotSessionNotFoundError(LookupError):
    pass


class PlotFigureNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class HighFrequencySignal:
    signal_id: str
    times: np.ndarray
    values: np.ndarray


class PlotSignalReader(Protocol):
    def read(self, source_path: Path, signals: Sequence[SignalInfo]) -> list[HighFrequencySignal]:
        ...


class EdfPlotSignalReader:
    """Load selected full-resolution signals into NumPy, not Python lists."""

    def read(self, source_path: Path, signals: Sequence[SignalInfo]) -> list[HighFrequencySignal]:
        reader = pyedflib.EdfReader(str(source_path))
        try:
            labels = [label.strip() for label in reader.getSignalLabels()]
            counts = reader.getNSamples()
            axes = {}
            loaded = []
            for signal in signals:
                try:
                    index = labels.index(signal.label.strip())
                except ValueError as exc:
                    raise SignalNotFoundError(signal.id) from exc
                count = int(counts[index])
                sample_rate = float(reader.getSampleFrequency(index))
                if count <= 0 or not np.isfinite(sample_rate) or sample_rate <= 0:
                    raise ValueError("Signal has no samples or an invalid sample rate.")
                key = (count, sample_rate)
                if key not in axes:
                    times = np.arange(count, dtype=np.float64)
                    times /= sample_rate
                    axes[key] = times
                values = np.empty(count, dtype=np.float64)
                for start, chunk in _signal_chunks(reader, index, 0, count):
                    values[start:start + len(chunk)] = chunk
                loaded.append(HighFrequencySignal(signal.id, axes[key], values))
            return loaded
        finally:
            reader.close()


def _create_resampler(signals: Sequence[SignalInfo], max_points: int):
    try:
        from plotly.subplots import make_subplots
        from plotly_resampler import FigureResampler
    except ImportError as exc:
        raise PlotlyUnavailableError(
            "Install optional plot dependencies: pip install -r sleepwalker/server/requirements-plotly.txt"
        ) from exc
    figure = FigureResampler(
        make_subplots(rows=len(signals), cols=1, shared_xaxes=True, subplot_titles=[signal.label for signal in signals]),
        default_n_shown_samples=max_points,
    )
    for row, signal in enumerate(signals, start=1):
        figure.update_yaxes(title_text=signal.unit, row=row, col=1)
    figure.update_xaxes(title_text="Time (s)", row=len(signals), col=1)
    return figure


@dataclass
class PlotView:
    figure: Any
    lock: Any = field(default_factory=RLock)


class PlotlyService:
    def __init__(
        self, session_service: SessionService, *, signal_reader: PlotSignalReader | None = None,
        figure_factory: Callable = _create_resampler,
    ):
        self.session_service = session_service
        self.signal_reader = signal_reader if signal_reader is not None else EdfPlotSignalReader()
        self.figure_factory = figure_factory
        self._views: dict[tuple[str, str], PlotView] = {}
        self._pending: dict[tuple[str, str], object] = {}
        self._lock = RLock()

    def _get_session(self, session_id: str) -> Session:
        session = self.session_service.get(session_id)
        if session is None:
            self.delete_session(session_id)
            raise PlotSessionNotFoundError(session_id)
        return session

    def create_figure(
        self, session_id: str, signal_ids: Sequence[str], *, view_id: str = "main", max_points: int = 2000,
    ) -> dict:
        if not isinstance(view_id, str) or not view_id.strip():
            raise ValueError("view_id must be a non-empty string.")
        if type(max_points) is not int or not 3 <= max_points <= 100_000:
            raise ValueError("Plot max_points must be between 3 and 100000.")
        if isinstance(signal_ids, str) or not signal_ids or any(not isinstance(value, str) for value in signal_ids):
            raise ValueError("Select one or more signal IDs.")
        if len(set(signal_ids)) != len(signal_ids):
            raise ValueError("Signal IDs must be unique.")
        session = self._get_session(session_id)
        if session.state != SessionState.READY or session.format != StudyFormat.EDF or len(session.source_files) != 1:
            raise ValueError("Plotting requires a ready, single-EDF session.")
        available = {signal.id: signal for signal in session.signals or []}
        try:
            signals = [available[signal_id] for signal_id in signal_ids]
        except KeyError as exc:
            raise SignalNotFoundError(exc.args[0]) from exc

        key, ticket = (session_id, view_id), object()
        with self._lock:
            self._pending[key] = ticket
        try:
            # Import optional dependencies before loading any full-resolution arrays.
            figure = self.figure_factory(signals, max_points)
            data = self.signal_reader.read(session.source_files[0], signals)
            if len(data) != len(signals):
                raise ValueError("Plot reader did not return all requested signals.")
            for row, (signal, samples) in enumerate(zip(signals, data), start=1):
                if samples.signal_id != signal.id or len(samples.times) != len(samples.values) or not len(samples.times):
                    raise ValueError("Plot reader returned inconsistent signal data.")
                figure.add_trace(
                    {"type": "scattergl", "name": signal.label, "mode": "lines"},
                    hf_x=samples.times, hf_y=samples.values, row=row, col=1,
                )
            initial = figure.to_plotly_json()
            with self._lock:
                # DELETE/close or a newer selection must not resurrect an old figure.
                if self._get_session(session_id) is not session:
                    raise PlotSessionNotFoundError(session_id)
                if self._pending.get(key) is not ticket:
                    raise PlotFigureNotFoundError(key)
                self._views[key] = PlotView(figure)
            return initial
        finally:
            with self._lock:
                if self._pending.get(key) is ticket:
                    del self._pending[key]

    def update_figure(self, session_id: str, relayout_data: dict | None, *, view_id: str = "main"):
        session = self._get_session(session_id)
        if relayout_data is not None and not isinstance(relayout_data, dict):
            raise ValueError("relayout_data must be a dictionary or None.")
        with self._lock:
            view = self._views.get((session_id, view_id))
        if view is None:
            raise PlotFigureNotFoundError((session_id, view_id))
        with view.lock:
            patch = view.figure.construct_update_data_patch(relayout_data)
            with self._lock:
                # Discard in-flight updates after deletion or figure replacement.
                if self.session_service.store.get(session_id) is not session:
                    raise PlotSessionNotFoundError(session_id)
                if self._views.get((session_id, view_id)) is not view:
                    raise PlotFigureNotFoundError((session_id, view_id))
            return patch

    def has_figure(self, session_id: str, *, view_id: str = "main") -> bool:
        with self._lock:
            return (session_id, view_id) in self._views

    def delete_figure(self, session_id: str, *, view_id: str = "main") -> bool:
        with self._lock:
            key = (session_id, view_id)
            pending = self._pending.pop(key, None)
            view = self._views.pop(key, None)
            return view is not None or pending is not None

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            keys = {key for key in self._views.keys() | self._pending.keys() if key[0] == session_id}
            for key in keys:
                self._views.pop(key, None)
                self._pending.pop(key, None)
            return len(keys)

    def clear(self) -> None:
        with self._lock:
            self._views.clear()
            self._pending.clear()

    def register_dash_callback(self, dash_app, *, graph_id: str, session_store_id: str, view_id: str = "main"):
        """Bind one graph to a dcc.Store containing the selected API session ID."""
        try:
            from dash import Input, Output, State, no_update
        except ImportError as exc:
            raise PlotlyUnavailableError("Install the optional plot dependencies to register Dash callbacks.") from exc

        @dash_app.callback(
            Output(graph_id, "figure", allow_duplicate=True), Input(graph_id, "relayoutData"),
            State(session_store_id, "data"), prevent_initial_call=True,
        )
        def update(relayout_data, session_id):
            if not isinstance(session_id, str) or not session_id:
                return no_update
            try:
                return self.update_figure(session_id, relayout_data, view_id=view_id)
            except (PlotSessionNotFoundError, PlotFigureNotFoundError):
                return no_update

        return update
