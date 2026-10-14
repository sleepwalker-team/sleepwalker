import hashlib
import json
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest
from pydantic import ValidationError

from sleepwalker.server.schemas.export import ExportRequest

from sleepwalker.server.services.annotation_service import AnnotationService
from sleepwalker.server.services.export_service import ExportService
from sleepwalker.server.services.job_executor import LocalJobExecutor
from sleepwalker.server.services.job_service import JobService
from sleepwalker.server.services.model_registry import ModelInfo, ModelRegistry
from sleepwalker.server.services.result_export_service import (
    ResultExportOptions,
    ResultExportService,
)
from sleepwalker.server.services.workspace_export_service import WorkspaceExportService
from sleepwalker.server.storage.temporary_storage import TemporaryStorage
from sleepwalker.server.stores.annotation_store import AnnotationStore
from sleepwalker.server.stores.export_store import ExportState, ExportStore
from sleepwalker.server.stores.job_store import JobState, JobStore
from sleepwalker.server.stores.prediction_store import PredictionKind, PredictionResult, PredictionStore
from sleepwalker.server.stores.session_store import Session, SessionState


def _session() -> Session:
    created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return Session(
        id="sess-test",
        created_at=created_at,
        expires_at=created_at + timedelta(hours=2),
        last_accessed_at=created_at,
        source_files=[Path("record.edf")],
        state=SessionState.READY,
        duration=60.0,
        recording_start=datetime(2025, 1, 1),
    )


def _service(tmp_path) -> ExportService:
    result_service = ResultExportService(
        PredictionStore(),
        AnnotationService(AnnotationStore()),
    )
    return ExportService(
        ExportStore(),
        result_service,
        TemporaryStorage(tmp_path / "uploads"),
        WorkspaceExportService(result_service, ModelRegistry()),
    )


def test_export_job_writes_json_and_becomes_ready(tmp_path) -> None:
    export_service = _service(tmp_path)
    job_service = JobService(JobStore())
    job = job_service.create("sess-test", "export")
    export = export_service.create(_session(), job.id)
    operation = partial(
        export_service.run,
        export,
        _session(),
        ResultExportOptions(),
    )

    completed_job = LocalJobExecutor(job_service).execute(
        job.id,
        operation,
        success_message="Export completed",
    )
    export_service.finalize(export.id, completed_job)

    assert completed_job.state == JobState.SUCCEEDED
    assert completed_job.message == "Export completed"
    assert export.state == ExportState.READY
    assert export.path is not None
    assert export.size == export.path.stat().st_size
    assert json.loads(export.path.read_text(encoding="utf-8"))["format"] == (
        "sleepwalker-results"
    )


def test_cancelled_export_does_not_leave_a_download(tmp_path) -> None:
    export_service = _service(tmp_path)
    job_service = JobService(JobStore())
    job = job_service.create("sess-test", "export")
    export = export_service.create(_session(), job.id)
    job_service.request_cancel(job.id)
    operation_was_called = False

    def operation(_report_progress, _check_cancelled) -> None:
        nonlocal operation_was_called
        operation_was_called = True

    cancelled_job = LocalJobExecutor(job_service).execute(job.id, operation)
    export_service.finalize(export.id, cancelled_job)

    assert operation_was_called is False
    assert cancelled_job.state == JobState.CANCELLED
    assert export.state == ExportState.CANCELLED
    assert export.path is None
    assert export.size is None


@pytest.mark.parametrize("kind", list(PredictionKind))
def test_workspace_contains_sources_results_history_and_checksums(tmp_path, kind) -> None:
    service = _service(tmp_path)
    session = _session()
    source = tmp_path / "record.edf"
    source.write_bytes(b"original study bytes")
    session.source_files = [source]
    service.workspace_export_service.model_registry.add(
        ModelInfo(
            id="sleep-model",
            name="Sleep Stage Model",
            task="sleep_stage",
            classes=("wake", "n1"),
            input_channels=("eeg",),
            capabilities=("classification",),
            output_resolution=30.0,
            output_kind=kind,
            package_path=tmp_path / "sleep-model.swmodel",
        )
    )
    times = {"start": [0.0], "end": [30.0]} if kind == PredictionKind.EVENT else {"time": [15.0]}
    service.result_export_service.prediction_store.add(
        PredictionResult(
            session_id=session.id,
            model_id="sleep-model",
            task="sleep_stage",
            classes=("wake", "n1"),
            resolution=30.0,
            frame=pd.DataFrame({**times, "prob__wake": [0.8], "prob__n1": [0.2]}),
            kind=kind,
        )
    )
    annotations = service.result_export_service.annotation_service
    edit = annotations.set_label(session.id, "sleep_stage", 0.0, "n1")
    reverted = annotations.revert(session.id, edit.id)
    annotations.set_label("sess-other", "sleep_stage", 0.0, "wake")
    job_service = JobService(JobStore())
    job = job_service.create(session.id, "export")
    export = service.create(session, job.id, format="sleepwalker", mode="workspace")
    operation = partial(service.run, export, session, ResultExportOptions())

    completed = LocalJobExecutor(job_service).execute(job.id, operation)
    service.finalize(export.id, completed)

    assert export.state == ExportState.READY
    assert export.path is not None
    assert export.path.name.endswith(".sleepwalker.zip")
    with ZipFile(export.path) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "study/metadata.json",
            "study/source/record.edf",
            "predictions/0000.json",
            "annotations/edits.json",
            "provenance/models.json",
        }
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format"] == "sleepwalker"
        assert manifest["format_version"] == 1
        assert manifest["study"]["source_files"] == ["study/source/record.edf"]
        assert manifest["prediction_files"] == {"sleep_stage": "predictions/0000.json"}
        assert manifest["tasks"] == ["sleep_stage"]
        assert set(manifest["checksums"]) == set(archive.namelist()) - {"manifest.json"}
        for name, checksum in manifest["checksums"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum

        assert archive.read("study/source/record.edf") == source.read_bytes()
        task = json.loads(archive.read("predictions/0000.json"))
        assert task["kind"] == kind.value
        assert task["model_id"] == "sleep-model"
        assert task["predictions"][0]["probabilities"] == {"wake": 0.8, "n1": 0.2}
        edits = json.loads(archive.read("annotations/edits.json"))
        assert [item["id"] for item in edits] == [edit.id, reverted.id]
        assert edits[1]["reverted_edit_id"] == edit.id
        models = json.loads(archive.read("provenance/models.json"))
        assert models[0]["id"] == "sleep-model"
        assert models[0]["name"] == "Sleep Stage Model"
        assert models[0]["input_channels"] == ["eeg"]
        assert models == manifest["models"]
        assert "package_path" not in models[0]


def test_workspace_cancel_during_copy_removes_partial_archive(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("sleepwalker.server.storage.temporary_storage.UPLOAD_CHUNK_SIZE", 4)
    service = _service(tmp_path)
    session = _session()
    source = tmp_path / "record.edf"
    source.write_bytes(b"original study bytes")
    session.source_files = [source]
    job_service = JobService(JobStore())
    job = job_service.create(session.id, "export")
    export = service.create(session, job.id, format="sleepwalker", mode="workspace")
    checks = 0

    def operation(report_progress, check_cancelled) -> None:
        def cancel_during_copy() -> None:
            nonlocal checks
            checks += 1
            if checks == 6:
                job_service.request_cancel(job.id)
            check_cancelled()

        service.run(export, session, ResultExportOptions(), report_progress, cancel_during_copy)

    cancelled = LocalJobExecutor(job_service).execute(job.id, operation)
    service.finalize(export.id, cancelled)

    assert cancelled.state == JobState.CANCELLED
    assert export.state == ExportState.CANCELLED
    assert list((service.storage.root / session.id / "exports").iterdir()) == []
    assert source.read_bytes() == b"original study bytes"


def test_workspace_failure_with_missing_source_has_safe_job_error(tmp_path) -> None:
    service = _service(tmp_path)
    session = _session()
    session.source_files = [tmp_path / "missing.edf"]
    job_service = JobService(JobStore())
    job = job_service.create(session.id, "export")
    export = service.create(session, job.id, format="sleepwalker", mode="workspace")
    operation = partial(service.run, export, session, ResultExportOptions())

    failed = LocalJobExecutor(job_service).execute(job.id, operation)
    service.finalize(export.id, failed)

    assert failed.state == JobState.FAILED
    assert failed.error is not None
    assert failed.error.code == "EXPORT_FAILED"
    assert str(tmp_path) not in failed.error.message
    assert export.state == ExportState.FAILED
    assert export.path is None


def test_export_finalization_tolerates_completed_session_cleanup(tmp_path) -> None:
    service = _service(tmp_path)
    session = _session()
    job_service = JobService(JobStore())
    job = job_service.create(session.id, "export")
    export = service.create(session, job.id)
    completed = LocalJobExecutor(job_service).execute(
        job.id, partial(service.run, export, session, ResultExportOptions()),
    )
    service.storage.delete_session(session.id)
    service.store.delete_session(session.id)

    assert service.finalize(export.id, completed) is None
    assert not (service.storage.root / session.id).exists()


@pytest.mark.parametrize("payload", [
    {"format": "json", "mode": "workspace"},
    {"format": "sleepwalker", "mode": "results"},
    {"format": "sleepwalker", "mode": "workspace", "include": {"probabilities": False}},
])
def test_export_request_rejects_invalid_format_or_incomplete_workspace(payload) -> None:
    with pytest.raises(ValidationError):
        ExportRequest.model_validate(payload)
