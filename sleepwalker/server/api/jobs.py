import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from ..core.jobs import TERMINAL_STATES, Job, JobService, JobSubscription
from .deps import Jobs
from .errors import conflict, not_found
from .schemas import ErrorDetail, ErrorResponse, JobResponse

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def to_job_response(job: Job) -> JobResponse:
    error = ErrorDetail(code=job.error.code, message=job.error.message, details=job.error.details) if job.error is not None else None
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


def _job_not_found():
    return not_found("JOB_NOT_FOUND", "Job not found.")


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, jobs: Jobs) -> JobResponse:
    job = jobs.snapshot(job_id)
    if job is None:
        raise _job_not_found()
    return to_job_response(job)


async def stream_job_events(job_id: str, jobs: JobService, subscription: JobSubscription, heartbeat_seconds: float = 15.0) -> AsyncIterator[str]:
    try:
        while True:
            try:
                job = await asyncio.wait_for(subscription.get(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue

            if job is None:
                error = ErrorResponse(error=ErrorDetail(code="JOB_NOT_FOUND", message="Job was deleted.", details={"job_id": job_id}))
                yield f"event: job_deleted\ndata: {error.model_dump_json()}\n\n"
                return

            yield f"event: job_update\ndata: {to_job_response(job).model_dump_json()}\n\n"
            if job.state in TERMINAL_STATES:
                return
    finally:
        jobs.unsubscribe(job_id, subscription)


@router.get(
    "/{job_id}/events",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}},
        404: {"model": ErrorResponse},
    },
)
async def get_job_events(job_id: str, jobs: Jobs) -> StreamingResponse:
    try:
        subscription = jobs.subscribe(job_id)
    except KeyError as exc:
        raise _job_not_found() from exc

    return StreamingResponse(
        stream_job_events(job_id, jobs, subscription),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        # A client that disconnects before streaming starts never runs the generator's cleanup.
        background=BackgroundTask(jobs.unsubscribe, job_id, subscription),
    )


@router.post("/{job_id}/cancel", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def cancel_job(job_id: str, jobs: Jobs) -> JobResponse:
    try:
        return to_job_response(jobs.request_cancel(job_id))
    except KeyError as exc:
        raise _job_not_found() from exc
    except ValueError as exc:
        job = jobs.get(job_id)
        details = {"state": job.state.value} if job is not None else None
        raise conflict("INVALID_JOB_STATE", "Job cannot be cancelled in its current state.", details) from exc
