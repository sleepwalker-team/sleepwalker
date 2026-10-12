from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobPhase(str, Enum):
    LOADING = "loading"
    PREPROCESSING = "preprocessing"
    INFERENCE = "inference"
    POSTPROCESSING = "postprocessing"


@dataclass(frozen=True)
class JobError:
    code: str
    message: str
    details: dict[str, Any] | None = None


@dataclass
class Job:
    id: str
    session_id: str
    type: str
    state: JobState
    created_at: datetime
    updated_at: datetime

    phase: JobPhase | None = None
    progress: float | None = None
    message: str | None = None
    error: JobError | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = RLock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._add(job)

    def add_if_session_idle(self, job: Job) -> Job | None:
        with self._lock:
            active_job = next(
                (
                    stored_job
                    for stored_job in self._jobs.values()
                    if stored_job.session_id == job.session_id
                    and stored_job.state
                    in {
                        JobState.QUEUED,
                        JobState.RUNNING,
                        JobState.CANCELLING,
                    }
                ),
                None,
            )

            if active_job is not None:
                return active_job

            self._add(job)
            return None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_for_session(self, session_id: str) -> list[Job]:
        with self._lock:
            return [
                job
                for job in self._jobs.values()
                if job.session_id == session_id
            ]

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            job_ids = [
                job.id
                for job in self._jobs.values()
                if job.session_id == session_id
            ]

            for job_id in job_ids:
                del self._jobs[job_id]

            return len(job_ids)

    def _add(self, job: Job) -> None:
        if job.id in self._jobs:
            raise ValueError(f"Job ID is already registered: {job.id}")

        self._jobs[job.id] = job
