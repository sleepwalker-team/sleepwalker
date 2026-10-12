from fastapi import APIRouter, Depends, Request, status

from ..errors import APIError
from ..schemas.common import ErrorDetail
from ..schemas.job import JobResponse
from ..services.job_service import JobService
from ..stores.job_store import Job


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
    job = service.get(job_id)

    if job is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="JOB_NOT_FOUND",
            message="Job not found.",
        )

    return to_job_response(job)