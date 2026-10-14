from datetime import datetime, timedelta, timezone

import pandas as pd

from sleepwalker.server.api.sessions import execute_inference_job
from sleepwalker.server.services.annotation_service import AnnotationService
from sleepwalker.server.services.job_executor import LocalJobExecutor
from sleepwalker.server.services.job_service import JobService
from sleepwalker.server.services.session_cleanup_service import SessionCleanupService
from sleepwalker.server.services.session_service import SessionService
from sleepwalker.server.storage.temporary_storage import TemporaryStorage
from sleepwalker.server.stores.annotation_store import AnnotationStore
from sleepwalker.server.stores.export_store import Export, ExportState, ExportStore
from sleepwalker.server.stores.job_store import JobPhase, JobState, JobStore
from sleepwalker.server.stores.prediction_store import PredictionResult, PredictionStore
from sleepwalker.server.stores.session_store import SessionStore


def test_cleanup_removes_expired_session_but_keeps_session_with_active_job(
    tmp_path,
) -> None:
    created_at = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    cleanup_at = created_at + timedelta(hours=3)
    times = iter([created_at, created_at, cleanup_at])

    session_service = SessionService(
        SessionStore(),
        ttl=timedelta(hours=2),
        clock=lambda: next(times),
    )
    job_service = JobService(JobStore())
    prediction_store = PredictionStore()
    annotation_service = AnnotationService(AnnotationStore())
    storage = TemporaryStorage(tmp_path / "uploads")

    removable_session = session_service.create(source_files=[])
    protected_session = session_service.create(source_files=[])

    removable_directory = storage.root / removable_session.id
    removable_directory.mkdir()
    (removable_directory / "recording.edf").write_bytes(b"test")

    protected_directory = storage.root / protected_session.id
    protected_directory.mkdir()

    prediction_store.add(
        PredictionResult(
            session_id=removable_session.id,
            model_id="model-test",
            task="sleep_stage",
            classes=("wake", "n1"),
            resolution=30.0,
            frame=pd.DataFrame({"time": [15.0]}),
        )
    )
    annotation_service.set_label(
        session_id=removable_session.id,
        task="sleep_stage",
        start=0.0,
        label="wake",
    )

    completed_job = job_service.create(
        session_id=removable_session.id,
        job_type="inference",
    )
    job_service.mark_running(
        completed_job.id,
        phase=JobPhase.LOADING,
        message="Loading",
    )
    job_service.mark_succeeded(completed_job.id)

    job_service.create(
        session_id=protected_session.id,
        job_type="inference",
    )

    cleanup_service = SessionCleanupService(
        session_service=session_service,
        job_service=job_service,
        storage=storage,
        prediction_store=prediction_store,
        annotation_service=annotation_service,
        export_store=ExportStore(),
    )

    assert cleanup_service.cleanup_expired() == [removable_session.id]

    assert session_service.store.get(removable_session.id) is None
    assert job_service.get(completed_job.id) is None
    assert prediction_store.list_for_session(removable_session.id) == []
    assert annotation_service.store.list_for_session(removable_session.id) == []
    assert not removable_directory.exists()

    assert session_service.store.get(protected_session.id) is protected_session
    assert protected_directory.exists()


def test_requested_delete_waits_for_active_job_before_removing_resources(
    tmp_path,
) -> None:
    session_service = SessionService(SessionStore())
    job_service = JobService(JobStore())
    prediction_store = PredictionStore()
    annotation_service = AnnotationService(AnnotationStore())
    storage = TemporaryStorage(tmp_path / "uploads")
    session = session_service.create(source_files=[])
    session_directory = storage.root / session.id
    session_directory.mkdir()
    (session_directory / "recording.edf").write_bytes(b"test")
    job = job_service.create(session.id, "inference")
    cleanup_service = SessionCleanupService(
        session_service=session_service,
        job_service=job_service,
        storage=storage,
        prediction_store=prediction_store,
        annotation_service=annotation_service,
        export_store=ExportStore(),
    )

    assert cleanup_service.request_delete(session.id) is True
    assert session_service.store.get(session.id) is None
    assert job.state == JobState.CANCELLING
    assert session_directory.exists()

    job_service.mark_cancelled(job.id)

    assert cleanup_service.finalize_pending_deletion(session.id) is True
    assert job_service.get(job.id) is None
    assert not session_directory.exists()


def test_cancelled_job_finalizes_requested_session_delete(tmp_path) -> None:
    session_service = SessionService(SessionStore())
    job_service = JobService(JobStore())
    storage = TemporaryStorage(tmp_path / "uploads")
    session = session_service.create(source_files=[])
    session_directory = storage.root / session.id
    session_directory.mkdir()
    job = job_service.create(session.id, "inference")
    cleanup_service = SessionCleanupService(
        session_service=session_service,
        job_service=job_service,
        storage=storage,
        prediction_store=PredictionStore(),
        annotation_service=AnnotationService(AnnotationStore()),
        export_store=ExportStore(),
    )
    operation_was_called = False

    def operation(_report_progress, _check_cancelled) -> None:
        nonlocal operation_was_called
        operation_was_called = True

    cleanup_service.request_delete(session.id)
    execute_inference_job(
        job.id,
        session.id,
        operation,
        LocalJobExecutor(job_service),
        cleanup_service,
    )

    assert operation_was_called is False
    assert job_service.get(job.id) is None
    assert not session_directory.exists()


def test_cleanup_removes_export_metadata_and_file(tmp_path) -> None:
    session_service = SessionService(SessionStore())
    job_service = JobService(JobStore())
    storage = TemporaryStorage(tmp_path / "uploads")
    export_store = ExportStore()
    session = session_service.create(source_files=[])
    export_path = storage.write_json_export(session.id, "export-test", {"test": True})
    export_store.add(
        Export(
            id="export-test",
            session_id=session.id,
            job_id="job-test",
            state=ExportState.READY,
            format="json",
            mode="results",
            created_at=session.created_at,
            expires_at=session.expires_at,
            size=export_path.stat().st_size,
            path=export_path,
        )
    )
    cleanup_service = SessionCleanupService(
        session_service=session_service,
        job_service=job_service,
        storage=storage,
        prediction_store=PredictionStore(),
        annotation_service=AnnotationService(AnnotationStore()),
        export_store=export_store,
    )

    assert cleanup_service.delete_session(session.id) is True
    assert export_store.get("export-test") is None
    assert not export_path.exists()
