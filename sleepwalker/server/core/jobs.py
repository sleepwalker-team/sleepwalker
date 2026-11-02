"""Background jobs: state machine, execution and live updates for SSE clients."""

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock, RLock
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


ACTIVE_STATES = {JobState.QUEUED, JobState.RUNNING, JobState.CANCELLING}
TERMINAL_STATES = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


class JobPhase(str, Enum):
    LOADING = "loading"
    PREPROCESSING = "preprocessing"
    INFERENCE = "inference"
    POSTPROCESSING = "postprocessing"
    EXPORTING = "exporting"
    IMPORTING = "importing"


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


ProgressCallback = Callable[[JobPhase, float | None, str], None]
CancellationCheck = Callable[[], None]
JobOperation = Callable[[ProgressCallback, CancellationCheck], None]


class JobExecutionError(Exception):
    """An expected job failure whose code and message are safe to show clients."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class JobCancelledError(Exception):
    pass


class ActiveJobExistsError(Exception):
    def __init__(self, job: Job) -> None:
        super().__init__(f"Session already has an active job: {job.id}")
        self.job = job


class JobSubscription:
    """Bounded mailbox between job worker threads and one async consumer."""

    def __init__(self, initial: Job) -> None:
        self._loop = asyncio.get_running_loop()
        self._initial: Job | None = initial
        self._updates: deque[Job | None] = deque(maxlen=64)
        self._wakeup = asyncio.Event()
        self._lock = Lock()
        self._notification_pending = False
        self._closed = False

    def publish(self, job: Job | None) -> None:
        with self._lock:
            if self._closed:
                return

            self._updates.append(job)
            if self._notification_pending:
                return
            self._notification_pending = True

        try:
            self._loop.call_soon_threadsafe(self._wakeup.set)
        except RuntimeError:
            # The consumer's event loop has already shut down.
            self.close()

    async def get(self) -> Job | None:
        while True:
            with self._lock:
                if self._initial is not None:
                    initial = self._initial
                    self._initial = None
                    return initial
                if self._updates:
                    return self._updates.popleft()
                if self._closed:
                    return None

                self._wakeup.clear()
                self._notification_pending = False

            await self._wakeup.wait()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._initial = None
            self._updates.clear()

        try:
            self._loop.call_soon_threadsafe(self._wakeup.set)
        except RuntimeError:
            pass


class JobService:
    def __init__(self, on_completion: Callable[[str], None] | None = None) -> None:
        self.on_completion = on_completion
        self._jobs: dict[str, Job] = {}
        self._subscriptions: dict[str, set[JobSubscription]] = {}
        self._lock = RLock()

    def create(self, session_id: str, job_type: str, parameters: dict[str, Any] | None = None) -> Job:
        job = self._new_job(session_id, job_type, parameters)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def create_if_session_idle(self, session_id: str, job_type: str, parameters: dict[str, Any] | None = None) -> Job:
        with self._lock:
            active = self.list_active_jobs(session_id)
            if active:
                raise ActiveJobExistsError(active[0])
            return self.create(session_id, job_type, parameters)

    @staticmethod
    def _new_job(session_id: str, job_type: str, parameters: dict[str, Any] | None) -> Job:
        now = datetime.now(timezone.utc)
        return Job(
            id=f"job_{uuid4()}",
            session_id=session_id,
            type=job_type,
            state=JobState.QUEUED,
            created_at=now,
            updated_at=now,
            message="Waiting for execution",
            parameters=deepcopy(parameters) if parameters is not None else {},
        )

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> Job | None:
        with self._lock:
            return deepcopy(self._jobs.get(job_id))

    def list_active_jobs(self, session_id: str) -> list[Job]:
        with self._lock:
            return [job for job in self._jobs.values() if job.session_id == session_id and job.state in ACTIVE_STATES]

    def has_active_job(self, session_id: str) -> bool:
        return bool(self.list_active_jobs(session_id))

    def subscribe(self, job_id: str) -> JobSubscription:
        with self._lock:
            subscription = JobSubscription(deepcopy(self._require(job_id)))
            self._subscriptions.setdefault(job_id, set()).add(subscription)
            return subscription

    def unsubscribe(self, job_id: str, subscription: JobSubscription) -> None:
        with self._lock:
            subscribers = self._subscriptions.get(job_id)
            if subscribers is not None:
                subscribers.discard(subscription)
                if not subscribers:
                    del self._subscriptions[job_id]

            subscription.close()

    def mark_running(self, job_id: str, phase: JobPhase, message: str) -> Job:
        return self._transition(job_id, {JobState.QUEUED}, JobState.RUNNING, phase=phase, message=message)

    def update_progress(self, job_id: str, *, phase: JobPhase, progress: float | None, message: str) -> Job:
        with self._lock:
            job = self._require(job_id)
            if job.state != JobState.RUNNING:
                raise ValueError(f"Cannot update progress for {job.state.value} job {job.id}.")
            if progress is not None and not 0.0 <= progress <= 1.0:
                raise ValueError("Job progress must be between 0.0 and 1.0.")
            return self._apply(job, phase=phase, progress=progress, message=message)

    def mark_succeeded(self, job_id: str, message: str = "Inference completed") -> Job:
        return self._transition(job_id, {JobState.RUNNING}, JobState.SUCCEEDED, phase=None, progress=1.0, message=message)

    def mark_failed(self, job_id: str, error: JobError) -> Job:
        return self._transition(job_id, {JobState.RUNNING}, JobState.FAILED, message=error.message, error=error)

    def mark_cancelled(self, job_id: str) -> Job:
        return self._transition(job_id, {JobState.CANCELLING}, JobState.CANCELLED, phase=None, progress=None, message="Cancelled")

    def request_cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self._require(job_id)
            if job.state == JobState.CANCELLING:
                return job
            return self._transition(job_id, {JobState.QUEUED, JobState.RUNNING}, JobState.CANCELLING, message="Cancellation requested")

    def is_cancellation_requested(self, job_id: str) -> bool:
        with self._lock:
            return self._require(job_id).state == JobState.CANCELLING

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            job_ids = [job.id for job in self._jobs.values() if job.session_id == session_id]
            for job_id in job_ids:
                del self._jobs[job_id]
                for subscription in self._subscriptions.pop(job_id, ()):
                    subscription.publish(None)
            return len(job_ids)

    def _require(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job ID: {job_id}")
        return job

    def _transition(self, job_id: str, allowed: set[JobState], target: JobState, **changes: Any) -> Job:
        with self._lock:
            job = self._require(job_id)
            if job.state not in allowed:
                raise ValueError(f"Job {job.id} cannot transition from {job.state.value} to {target.value}.")
            # Renew the session before the job becomes terminal, so expiry
            # cleanup cannot remove its results in between.
            if target in TERMINAL_STATES and self.on_completion is not None:
                self.on_completion(job.session_id)
            return self._apply(job, state=target, **changes)

    def _apply(self, job: Job, **changes: Any) -> Job:
        for name, value in changes.items():
            setattr(job, name, value)
        job.updated_at = datetime.now(timezone.utc)
        for subscription in self._subscriptions.get(job.id, ()):
            subscription.publish(deepcopy(job))
        return job


class JobExecutor:
    """Runs one operation and maps its outcome onto the job's final state."""

    def __init__(self, job_service: JobService) -> None:
        self.job_service = job_service

    def execute(self, job_id: str, operation: JobOperation, success_message: str = "Inference completed") -> Job:
        jobs = self.job_service

        def check_cancelled() -> None:
            if jobs.is_cancellation_requested(job_id):
                raise JobCancelledError

        def report_progress(phase: JobPhase, progress: float | None, message: str) -> None:
            check_cancelled()
            jobs.update_progress(job_id, phase=phase, progress=progress, message=message)

        try:
            check_cancelled()
            jobs.mark_running(job_id, phase=JobPhase.LOADING, message="Loading resources")
            operation(report_progress, check_cancelled)
            check_cancelled()
            return jobs.mark_succeeded(job_id, success_message)
        except JobCancelledError:
            return jobs.mark_cancelled(job_id)
        except Exception as exc:
            # A cancellation wins over whatever failure it may have provoked.
            if jobs.is_cancellation_requested(job_id):
                return jobs.mark_cancelled(job_id)
            return jobs.mark_failed(job_id, _to_job_error(job_id, exc))


def _to_job_error(job_id: str, exc: Exception) -> JobError:
    if isinstance(exc, JobExecutionError):
        return JobError(code=exc.code, message=exc.message, details=exc.details)
    if isinstance(exc, MemoryError):
        return JobError(code="RESOURCE_EXHAUSTED", message="Inference could not be completed because compute memory was exhausted.")
    logger.exception("Unexpected error while executing job %s", job_id)
    return JobError(code="INTERNAL_ERROR", message="An unexpected error occurred while executing the job.")
