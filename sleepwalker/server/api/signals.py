import math

from fastapi import APIRouter, Depends, Request, status

from ..errors import APIError
from ..schemas.signal import (
    SignalDataResponse,
    SignalListResponse,
    SignalSummaryResponse,
)
from ..services.session_service import SessionService
from ..services.signal_reader import SignalNotFoundError, SignalReader, downsample_extrema


router = APIRouter(prefix="/api/v1/sessions", tags=["signals"])


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def get_signal_reader(request: Request) -> SignalReader:
    return request.app.state.signal_reader


@router.get("/{session_id}/signals", response_model=SignalListResponse)
def list_signals(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
) -> SignalListResponse:
    session = session_service.get(session_id)

    if session is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="SESSION_NOT_FOUND",
            message="Session not found.",
        )

    return SignalListResponse(
        signals=[
            SignalSummaryResponse(
                id=signal.id,
                label=signal.label,
                unit=signal.unit,
                sample_rate=signal.sample_rate,
                start=0.0,
                end=session.duration or 0.0,
            )
            for signal in session.signals or []
        ]
    )


@router.get("/{session_id}/signals/{signal_id}", response_model=SignalDataResponse)
def get_signal_data(
    session_id: str,
    signal_id: str,
    start: float = 0.0,
    end: float | None = None,
    max_points: int = 2000,
    session_service: SessionService = Depends(get_session_service),
    signal_reader: SignalReader = Depends(get_signal_reader),
) -> SignalDataResponse:
    session = session_service.get(session_id)

    if session is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="SESSION_NOT_FOUND",
            message="Session not found.",
        )

    signal = next(
        (
            candidate
            for candidate in session.signals or []
            if candidate.id == signal_id
        ),
        None,
    )

    if signal is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="SIGNAL_NOT_FOUND",
            message="Signal not found.",
            details={"signal_id": signal_id},
        )

    if max_points < 2 or max_points > 100_000:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_MAX_POINTS",
            message="max_points must be between 2 and 100000.",
        )

    if session.duration is None or not session.source_files:
        raise APIError(
            status_code=status.HTTP_409_CONFLICT,
            code="INVALID_SESSION_STATE",
            message="Session is not ready for signal access.",
        )

    requested_end = session.duration if end is None else end

    if (
        not math.isfinite(start)
        or not math.isfinite(requested_end)
        or requested_end <= start
    ):
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_TIME_RANGE",
            message="Signal time range is invalid.",
            details={"start": start, "end": requested_end},
        )

    range_start = max(0.0, start)
    range_end = min(session.duration, requested_end)

    if range_start >= range_end:
        raise APIError(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            code="RANGE_NOT_SATISFIABLE",
            message="Signal time range does not overlap the recording.",
            details={"start": start, "end": requested_end},
        )

    try:
        samples = signal_reader.read(
            session.source_files[0], signal.id, range_start, range_end, max_points=max_points,
        )
    except SignalNotFoundError as exc:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="SIGNAL_NOT_FOUND",
            message="Signal not found.",
            details={"signal_id": signal_id},
        ) from exc

    if not samples.times:
        raise APIError(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            code="RANGE_NOT_SATISFIABLE",
            message="No signal samples exist in the requested range.",
        )

    reduced = downsample_extrema(samples, max_points)

    return SignalDataResponse(
        signal=signal.id,
        unit=signal.unit,
        sample_rate=signal.sample_rate,
        start=range_start,
        end=range_end,
        downsampled=(reduced.sample_count or len(samples.times)) > len(reduced.times),
        points=list(zip(reduced.times, reduced.values)),
    )
