import math

from fastapi import APIRouter, status

from ..recording.signals import MAX_POINTS_LIMIT, MIN_POINTS, SignalNotFoundError
from .deps import SessionDep, SignalReader
from .errors import APIError, bad_request, conflict, not_found
from .schemas import SignalDataResponse, SignalListResponse, SignalSummaryResponse

router = APIRouter(prefix="/api/v1/sessions", tags=["signals"])


@router.get("/{session_id}/signals", response_model=SignalListResponse)
def list_signals(session: SessionDep) -> SignalListResponse:
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
def get_signal_data(session: SessionDep, signal_id: str, reader: SignalReader, start: float = 0.0, end: float | None = None, max_points: int = 2000) -> SignalDataResponse:
    signal_not_found = not_found("SIGNAL_NOT_FOUND", "Signal not found.", {"signal_id": signal_id})
    signal = next((candidate for candidate in session.signals or [] if candidate.id == signal_id), None)
    if signal is None:
        raise signal_not_found

    if not MIN_POINTS <= max_points <= MAX_POINTS_LIMIT:
        raise bad_request(f"max_points must be between {MIN_POINTS} and {MAX_POINTS_LIMIT}.", code="INVALID_MAX_POINTS")

    if session.duration is None or not session.source_files:
        raise conflict("INVALID_SESSION_STATE", "Session is not ready for signal access.")

    requested_end = session.duration if end is None else end
    if not math.isfinite(start) or not math.isfinite(requested_end) or requested_end <= start:
        raise bad_request("Signal time range is invalid.", {"start": start, "end": requested_end}, code="INVALID_TIME_RANGE")

    range_start = max(0.0, start)
    range_end = min(session.duration, requested_end)
    if range_start >= range_end:
        raise APIError(
            status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE, "RANGE_NOT_SATISFIABLE",
            "Signal time range does not overlap the recording.", {"start": start, "end": requested_end},
        )

    try:
        samples = reader.read(session.source_files[0], signal.id, range_start, range_end, max_points=max_points)
    except SignalNotFoundError as exc:
        raise signal_not_found from exc

    if not samples.times:
        raise APIError(status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE, "RANGE_NOT_SATISFIABLE", "No signal samples exist in the requested range.")

    return SignalDataResponse(
        signal=signal.id,
        unit=signal.unit,
        sample_rate=signal.sample_rate,
        start=range_start,
        end=range_end,
        downsampled=samples.sample_count > len(samples.times),
        points=list(zip(samples.times, samples.values)),
    )
