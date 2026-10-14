import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from ..errors import APIError
from ..schemas.common import ErrorDetail, ErrorResponse
from ..schemas.job import JobResponse
from ..services.job_events import JobSubscription
from ..services.job_service import JobService
from ..stores.job_store import Job, JobState


router = APIRouter(
    prefix="/api/v1/jobs",
    tags=["jobs"],
)


def get_job_service(request: Request) -> JobService:
    return request.app.state.job_service


def to_job_response(job: Job) -> JobResponse:
    error = None

    if job.error is not None:
        error = ErrorDetail(
            code=job.error.code,
            message=job.error.message,
            details=job.error.details,
        )

    return JobResponse(
        id=job.id,
        type=job.type,
        state=job.state,
        phase=job.phase,
        progress=job.progress,
        message=job.message,
        updated_at=job.updated_at,
        error=error,
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
)
def get_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> JobResponse:
    job = service.snapshot(job_id)

    if job is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="JOB_NOT_FOUND",
            message="Job not found.",
        )

    return to_job_response(job)


async def stream_job_events(
    job_id: str,
    service: JobService,
    subscription: JobSubscription,
    heartbeat_seconds: float = 15.0,
) -> AsyncIterator[str]:
    try:
        while True:
            try:
                job = await asyncio.wait_for(subscription.get(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue

            if job is None:
                error = ErrorResponse(
                    error=ErrorDetail(code="JOB_NOT_FOUND", message="Job was deleted.", details={"job_id": job_id}),
                )
                yield f"event: job_deleted\ndata: {error.model_dump_json()}\n\n"
                return

            data = to_job_response(job).model_dump_json()
            yield f"event: job_update\ndata: {data}\n\n"
            if job.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
                return
    finally:
        service.unsubscribe(job_id, subscription)


@router.get(
    "/{job_id}/events",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}},
        404: {"model": ErrorResponse},
    },
)
async def get_job_events(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> StreamingResponse:
    try:
        subscription = service.subscribe(job_id)
    except KeyError as exc:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="JOB_NOT_FOUND",
            message="Job not found.",
        ) from exc

    return StreamingResponse(
        stream_job_events(job_id, service, subscription),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(service.unsubscribe, job_id, subscription),
    )


@router.post(
    "/{job_id}/cancel",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def cancel_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> JobResponse:
    try:
        job = service.request_cancel(job_id)
    except KeyError as exc:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="JOB_NOT_FOUND",
            message="Job not found.",
        ) from exc
    except ValueError as exc:
        job = service.get(job_id)
        details = {"state": job.state.value} if job is not None else None
        raise APIError(
            status_code=status.HTTP_409_CONFLICT,
            code="INVALID_JOB_STATE",
            message="Job cannot be cancelled in its current state.",
            details=details,
        ) from exc

    return to_job_response(job)
