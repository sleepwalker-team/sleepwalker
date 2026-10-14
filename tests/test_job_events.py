import asyncio
from threading import Thread

import pytest

from sleepwalker.server.api.jobs import stream_job_events
from sleepwalker.server.services.job_service import JobService
from sleepwalker.server.stores.job_store import JobError, JobPhase, JobState, JobStore


def test_subscription_preserves_initial_state_and_update_snapshots() -> None:
    async def scenario():
        service = JobService(JobStore())
        job = service.create("sess-test", "inference")
        subscription = service.subscribe(job.id)
        service.mark_running(job.id, JobPhase.LOADING, "Loading model")
        service.update_progress(job.id, phase=JobPhase.INFERENCE, progress=0.42, message="Running")
        service.mark_succeeded(job.id)

        snapshots = [await subscription.get() for _ in range(4)]
        assert [snapshot.state for snapshot in snapshots] == [
            JobState.QUEUED, JobState.RUNNING, JobState.RUNNING, JobState.SUCCEEDED,
        ]
        assert snapshots[1].phase == JobPhase.LOADING
        assert snapshots[2].progress == 0.42
        assert snapshots[0].state == JobState.QUEUED
        assert snapshots[-1] is not job
        service.unsubscribe(job.id, subscription)

    asyncio.run(scenario())


def test_worker_thread_wakes_waiting_subscription() -> None:
    async def scenario():
        service = JobService(JobStore())
        job = service.create("sess-test", "export")
        subscription = service.subscribe(job.id)
        await subscription.get()
        waiting = asyncio.create_task(subscription.get())
        await asyncio.sleep(0)
        worker = Thread(target=service.mark_running, args=(job.id, JobPhase.EXPORTING, "Exporting"))
        worker.start()
        try:
            update = await asyncio.wait_for(waiting, timeout=1.0)
            assert update.phase == JobPhase.EXPORTING
            assert update.state == JobState.RUNNING
        finally:
            worker.join(timeout=1.0)
            service.unsubscribe(job.id, subscription)

    asyncio.run(scenario())


def test_multiple_clients_and_jobs_have_independent_subscriptions() -> None:
    async def scenario():
        service = JobService(JobStore())
        job = service.create("sess-a", "inference")
        other_job = service.create("sess-b", "import")
        first = service.subscribe(job.id)
        second = service.subscribe(job.id)
        other = service.subscribe(other_job.id)
        for subscription in (first, second, other):
            await subscription.get()

        service.mark_running(job.id, JobPhase.INFERENCE, "Running")
        first_update = await first.get()
        second_update = await second.get()
        assert first_update.state == second_update.state == JobState.RUNNING
        assert first_update is not second_update
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(other.get(), timeout=0.01)

        service.unsubscribe(job.id, first)
        service.request_cancel(job.id)
        assert (await second.get()).state == JobState.CANCELLING
        service.unsubscribe(job.id, second)
        service.unsubscribe(other_job.id, other)
        assert service._subscriptions == {}

    asyncio.run(scenario())


def test_slow_client_buffer_is_bounded_and_preserves_terminal_state() -> None:
    async def scenario():
        service = JobService(JobStore())
        job = service.create("sess-test", "inference")
        subscription = service.subscribe(job.id)
        service.mark_running(job.id, JobPhase.INFERENCE, "Running")
        for step in range(1000):
            service.update_progress(job.id, phase=JobPhase.INFERENCE, progress=step / 1000, message="Running")
        service.mark_succeeded(job.id)

        updates = []
        while True:
            update = await asyncio.wait_for(subscription.get(), timeout=1.0)
            updates.append(update)
            if update.state == JobState.SUCCEEDED:
                break

        assert len(updates) == 65  # Initial snapshot plus at most 64 pending updates.
        assert updates[0].state == JobState.QUEUED
        assert updates[-1].progress == 1.0
        service.unsubscribe(job.id, subscription)

    asyncio.run(scenario())


def test_cancellation_is_published_once_and_failure_details_are_copied() -> None:
    async def scenario():
        service = JobService(JobStore())
        cancelled = service.create("sess-a", "inference")
        subscription = service.subscribe(cancelled.id)
        await subscription.get()
        service.request_cancel(cancelled.id)
        service.request_cancel(cancelled.id)
        service.mark_cancelled(cancelled.id)
        assert (await subscription.get()).state == JobState.CANCELLING
        assert (await subscription.get()).state == JobState.CANCELLED
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(subscription.get(), timeout=0.01)
        service.unsubscribe(cancelled.id, subscription)

        failed = service.create("sess-b", "import")
        subscription = service.subscribe(failed.id)
        service.mark_running(failed.id, JobPhase.IMPORTING, "Importing")
        error = JobError("INVALID_BUNDLE", "Invalid workspace", {"filename": "original.zip"})
        service.mark_failed(failed.id, error)
        error.details["filename"] = "changed.zip"
        await subscription.get()
        await subscription.get()
        snapshot = await subscription.get()
        assert snapshot.error.details == {"filename": "original.zip"}
        service.unsubscribe(failed.id, subscription)

    asyncio.run(scenario())


def test_deleted_job_wakes_subscribers_and_removes_registrations() -> None:
    async def scenario():
        service = JobService(JobStore())
        job = service.create("sess-test", "inference")
        subscription = service.subscribe(job.id)
        await subscription.get()
        assert service.delete_session("sess-test") == 1
        assert await asyncio.wait_for(subscription.get(), timeout=1.0) is None
        assert service._subscriptions == {}
        with pytest.raises(KeyError):
            service.subscribe(job.id)
        service.unsubscribe(job.id, subscription)

    asyncio.run(scenario())


def test_stream_heartbeats_and_client_close_release_subscription() -> None:
    async def scenario():
        service = JobService(JobStore())
        job = service.create("sess-test", "export")
        subscription = service.subscribe(job.id)
        stream = stream_job_events(job.id, service, subscription, heartbeat_seconds=0.005)
        try:
            assert (await stream.__anext__()).startswith("event: job_update\n")
            assert await stream.__anext__() == ": keep-alive\n\n"
        finally:
            await stream.aclose()
        assert service._subscriptions == {}

    asyncio.run(scenario())


def test_stream_sends_deletion_event_and_closes() -> None:
    async def scenario():
        service = JobService(JobStore())
        job = service.create("sess-test", "inference")
        subscription = service.subscribe(job.id)
        stream = stream_job_events(job.id, service, subscription)
        await stream.__anext__()
        service.delete_session("sess-test")
        assert (await stream.__anext__()).startswith("event: job_deleted\n")
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        assert service._subscriptions == {}

    asyncio.run(scenario())
