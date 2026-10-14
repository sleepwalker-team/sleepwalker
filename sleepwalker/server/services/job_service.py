from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from ..stores.job_store import Job, JobError, JobPhase, JobState, JobStore
from .job_events import JobSubscription


class ActiveJobExistsError(Exception):
    def __init__(self, job: Job) -> None:
        super().__init__(f"Session already has an active job: {job.id}")
        self.job = job


class JobService:
    def __init__(self, store: JobStore) -> None:
        self.store = store
        self._lock = RLock()
        self._subscriptions: dict[str, set[JobSubscription]] = {}

    def create(
        self,
        session_id: str,
        job_type: str,
        parameters: dict[str, Any] | None = None,
    ) -> Job:
        job = self._new_job(session_id, job_type, parameters)
        with self._lock:
            self.store.add(job)
        return job

    def create_if_session_idle(
        self,
        session_id: str,
        job_type: str,
        parameters: dict[str, Any] | None = None,
    ) -> Job:
        job = self._new_job(session_id, job_type, parameters)
        with self._lock:
            active_job = self.store.add_if_session_idle(job)

        if active_job is not None:
            raise ActiveJobExistsError(active_job)

        return job

    @staticmethod
    def _new_job(
        session_id: str,
        job_type: str,
        parameters: dict[str, Any] | None,
    ) -> Job:
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
            return self.store.get(job_id)

    def snapshot(self, job_id: str) -> Job | None:
        with self._lock:
            return deepcopy(self.store.get(job_id))

    def subscribe(self, job_id: str) -> JobSubscription:
        with self._lock:
            job = self._get_required(job_id)
            subscription = JobSubscription(deepcopy(job))
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

    def _publish(self, job: Job) -> None:
        for subscription in self._subscriptions.get(job.id, ()):
            subscription.publish(deepcopy(job))

    def mark_running(self, job_id: str, phase: JobPhase, message: str) -> Job:
        with self._lock:
            job = self._get_required(job_id)

            if job.state != JobState.QUEUED:
                self._raise_invalid_transition(job, JobState.RUNNING)

            job.state = JobState.RUNNING
            job.phase = phase
            job.message = message
            job.updated_at = datetime.now(timezone.utc)
            self._publish(job)

            return job

    def update_progress(
        self,
        job_id: str,
        *,
        phase: JobPhase,
        progress: float | None,
        message: str,
    ) -> Job:
        with self._lock:
            job = self._get_required(job_id)
            if job.state != JobState.RUNNING:
                raise ValueError(f"Cannot update progress for {job.state.value} job {job.id}.")

            if progress is not None and (
                isinstance(progress, bool)
                or not isinstance(progress, (int, float))
                or not 0.0 <= float(progress) <= 1.0
            ):
                raise ValueError("Job progress must be between 0.0 and 1.0.")

            job.phase = phase
            job.progress = float(progress) if progress is not None else None
            job.message = message
            job.updated_at = datetime.now(timezone.utc)
            self._publish(job)

            return job

    def mark_succeeded(self, job_id: str, message: str = "Inference completed") -> Job:
        with self._lock:
            job = self._get_required(job_id)

            if job.state != JobState.RUNNING:
                self._raise_invalid_transition(job, JobState.SUCCEEDED)

            job.state = JobState.SUCCEEDED
            job.phase = None
            job.progress = 1.0
            job.message = message
            job.updated_at = datetime.now(timezone.utc)
            self._publish(job)

            return job

    def mark_failed(self, job_id: str, error: JobError) -> Job:
        with self._lock:
            job = self._get_required(job_id)

            if job.state != JobState.RUNNING:
                self._raise_invalid_transition(job, JobState.FAILED)

            job.state = JobState.FAILED
            job.message = error.message
            job.error = deepcopy(error)
            job.updated_at = datetime.now(timezone.utc)
            self._publish(job)

            return job

    def request_cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self._get_required(job_id)

            if job.state == JobState.CANCELLING:
                return job

            if job.state not in {JobState.QUEUED, JobState.RUNNING}:
                self._raise_invalid_transition(job, JobState.CANCELLING)

            job.state = JobState.CANCELLING
            job.message = "Cancellation requested"
            job.updated_at = datetime.now(timezone.utc)
            self._publish(job)

            return job

    def mark_cancelled(self, job_id: str) -> Job:
        with self._lock:
            job = self._get_required(job_id)

            if job.state != JobState.CANCELLING:
                self._raise_invalid_transition(job, JobState.CANCELLED)

            job.state = JobState.CANCELLED
            job.phase = None
            job.progress = None
            job.message = "Cancelled"
            job.updated_at = datetime.now(timezone.utc)
            self._publish(job)

            return job

    def is_cancellation_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._get_required(job_id)
            return job.state == JobState.CANCELLING

    def list_active_jobs(self, session_id: str) -> list[Job]:
        with self._lock:
            return [
                job for job in self.store.list_for_session(session_id)
                if job.state in {JobState.QUEUED, JobState.RUNNING, JobState.CANCELLING}
            ]

    def _get_required(self, job_id: str) -> Job:
        job = self.store.get(job_id)

        if job is None:
            raise KeyError(f"Unknown job ID: {job_id}")

        return job

    @staticmethod
    def _raise_invalid_transition(job: Job, target: JobState) -> None:
        raise ValueError(
            f"Job {job.id} cannot transition "
            f"from {job.state.value} to {target.value}."
        )

    def has_active_job(self, session_id: str) -> bool:
        return bool(self.list_active_jobs(session_id))

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            jobs = self.store.list_for_session(session_id)
            count = self.store.delete_session(session_id)
            for job in jobs:
                for subscription in self._subscriptions.pop(job.id, ()):
                    subscription.publish(None)

            return count
