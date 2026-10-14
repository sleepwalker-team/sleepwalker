import hashlib
import json
import shutil
import stat
from datetime import timedelta
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sleepwalker.server.app import create_app
from sleepwalker.server.api.imports import execute_import_job
from sleepwalker.server.services.model_registry import ModelRegistry
from sleepwalker.server.services.workspace_export_service import WorkspaceExportService
from sleepwalker.server.storage.temporary_storage import TemporaryStorage
from sleepwalker.server.stores.job_store import JobState
from sleepwalker.server.stores.prediction_store import PredictionKind, PredictionResult
from sleepwalker.server.stores.session_store import SessionState
from sleepwalker.server.validation import validate_study
from sleepwalker.server.validation.workspace import (
    InvalidWorkspaceError,
    WorkspaceLimits,
    inspect_workspace,
)

DATA_DIR = Path(__file__).parent / "data"


def _bundle(tmp_path, kind=PredictionKind.EPOCH):
    storage = TemporaryStorage(tmp_path / "export")
    app = create_app(temporary_storage=storage)
    session = app.state.session_service.create([])
    study = validate_study([DATA_DIR / "signals_01.edf"])
    session.state = SessionState.READY
    session.source_files = study.source_files
    session.format = study.format
    session.duration = study.duration
    session.recording_start = study.recording_start
    session.signals = study.signals
    times = {"start": [12.0], "end": [18.0]} if kind == PredictionKind.EVENT else {"time": [15.0]}
    task = "sleep_stage" if kind == PredictionKind.EPOCH else "arousal"
    result = PredictionResult(
        session_id=session.id,
        model_id="model-test",
        task=task,
        classes=("negative", "positive"),
        resolution=30.0 if kind == PredictionKind.EPOCH else 1.0,
        kind=kind,
        frame=pd.DataFrame({**times, "prob__negative": [0.1], "prob__positive": [0.9]}),
    )
    app.state.prediction_store.add(result)
    annotations = app.state.annotation_service
    if kind == PredictionKind.EPOCH:
        edit = annotations.set_label(session.id, task, 0.0, "negative")
        annotations.revert(session.id, edit.id)
    else:
        edit = annotations.add_event(session.id, task, 12.0, 18.0, "positive")
        deletion = annotations.delete_event(session.id, edit.event_id)
        annotations.revert(session.id, deletion.id)

    exporter = WorkspaceExportService(app.state.result_export_service, ModelRegistry())
    documents, sources = exporter.build_documents(session)
    path = storage.write_workspace_export(session.id, "export-test", documents, sources, lambda: None)
    return path, session, app.state.result_export_service.build_payload(session)


def _alter_bundle(path, target, change, update_checksums=True):
    with ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    change(contents)
    if update_checksums:
        manifest = json.loads(contents["manifest.json"])
        manifest["checksums"] = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in contents.items() if name != "manifest.json"
        }
        contents["manifest.json"] = json.dumps(manifest).encode()
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    return target


def _edit_json(contents, name, change):
    document = json.loads(contents[name])
    change(document)
    contents[name] = json.dumps(document).encode()


def _prepare_import(tmp_path, bundle):
    app = create_app(temporary_storage=TemporaryStorage(tmp_path / "import"))
    session = app.state.session_service.create([])
    session.state = SessionState.PROCESSING
    directory = app.state.temporary_storage.root / session.id
    directory.mkdir()
    path = directory / "workspace.zip"
    shutil.copyfile(bundle, path)
    app.state.workspace_import_service.inspect(path)
    job = app.state.job_service.create(session.id, "import")
    return app, session, job, path


def _run_import(app, session, job, path):
    def operation(report_progress, check_cancelled):
        app.state.workspace_import_service.run(session, path, report_progress, check_cancelled)

    execute_import_job(
        session, job.id, operation, app.state.job_executor,
        app.state.workspace_import_service, app.state.session_cleanup_service,
    )


@pytest.mark.parametrize("kind", list(PredictionKind))
def test_workspace_round_trip_restores_results_and_remaps_annotation_references(tmp_path, kind):
    bundle, original, payload = _bundle(tmp_path, kind)
    app, session, job, path = _prepare_import(tmp_path, bundle)

    _run_import(app, session, job, path)

    assert job.state == JobState.SUCCEEDED
    assert job.message == "Import completed"
    assert session.state == SessionState.READY
    assert session.id != original.id
    assert session.created_at > original.created_at
    assert session.recording_start == original.recording_start
    assert session.source_files[0].read_bytes() == original.source_files[0].read_bytes()
    assert not path.exists()
    restored = app.state.result_export_service.build_payload(session)
    assert restored["tasks"] == payload["tasks"]
    original_ids = {edit["id"] for edit in payload["annotations"]}
    new_edits = restored["annotations"]
    assert original_ids.isdisjoint(edit["id"] for edit in new_edits)
    assert {edit["session_id"] for edit in new_edits} == {session.id}
    assert new_edits[-1]["reverted_edit_id"] == new_edits[-2]["id"]
    if kind == PredictionKind.EPOCH:
        assert app.state.annotation_service.current_labels(session.id) == []
    else:
        assert new_edits[0]["event_id"] == new_edits[1]["event_id"]
        assert new_edits[0]["event_id"] != payload["annotations"][0]["event_id"]
        assert len(app.state.annotation_service.current_events(session.id)) == 1
    assert app.state.model_registry.list() == []


@pytest.mark.parametrize("mutation", [
    "checksum", "source-metadata", "probabilities", "revert-reference", "foreign-session",
])
def test_invalid_workspace_data_fails_job_and_removes_restored_data(tmp_path, mutation):
    bundle, _, _ = _bundle(tmp_path)

    def change(contents):
        if mutation == "checksum":
            contents["study/source/signals_01.edf"] = b"corrupt recording"
        elif mutation == "source-metadata":
            _edit_json(contents, "study/metadata.json", lambda data: data.update(duration=12.0))
        elif mutation == "probabilities":
            _edit_json(
                contents, "predictions/0000.json",
                lambda data: data["predictions"][0]["probabilities"].pop("positive"),
            )
        elif mutation == "revert-reference":
            _edit_json(
                contents, "annotations/edits.json",
                lambda data: data[-1].update(reverted_edit_id="edit-missing"),
            )
        else:
            _edit_json(
                contents, "annotations/edits.json",
                lambda data: data[0].update(session_id="sess-other"),
            )

    altered = _alter_bundle(bundle, tmp_path / "altered.zip", change, mutation != "checksum")
    app, session, job, path = _prepare_import(tmp_path, altered)
    _run_import(app, session, job, path)

    assert job.state == JobState.FAILED
    assert job.error.code == "INVALID_BUNDLE"
    assert session.state == SessionState.FAILED
    assert session.source_files == []
    assert app.state.prediction_store.list_for_session(session.id) == []
    assert app.state.annotation_service.list_edits(session.id) == []
    assert not (app.state.temporary_storage.root / session.id).exists()


@pytest.mark.parametrize("name", [
    "../outside.edf", "/outside.edf", "study/../outside.edf", "study\\source\\x.edf",
])
def test_workspace_rejects_unsafe_archive_paths(tmp_path, name):
    bundle, _, _ = _bundle(tmp_path)
    altered = _alter_bundle(
        bundle, tmp_path / "altered.zip", lambda contents: contents.update({name: b"unsafe"}),
    )
    with pytest.raises(InvalidWorkspaceError):
        inspect_workspace(altered, WorkspaceLimits())
    assert not (tmp_path / "outside.edf").exists()


def test_workspace_rejects_symlink_entry(tmp_path):
    path = tmp_path / "symlink.zip"
    entry = ZipInfo("study/source/link.edf")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(path, "w") as archive:
        archive.writestr(entry, "../../outside.edf")
    with pytest.raises(InvalidWorkspaceError):
        inspect_workspace(path, WorkspaceLimits())


def test_workspace_rejects_excessive_uncompressed_size(tmp_path):
    bundle, _, _ = _bundle(tmp_path)
    with pytest.raises(InvalidWorkspaceError, match="uncompressed size"):
        inspect_workspace(bundle, WorkspaceLimits(max_total_bytes=1024))


def test_workspace_rejects_missing_member(tmp_path):
    bundle, _, _ = _bundle(tmp_path)
    altered = _alter_bundle(
        bundle, tmp_path / "missing.zip", lambda contents: contents.pop("annotations/edits.json"),
    )
    with pytest.raises(InvalidWorkspaceError):
        inspect_workspace(altered, WorkspaceLimits())


def test_cancelled_import_cleans_up_partially_extracted_sources(tmp_path):
    bundle, _, _ = _bundle(tmp_path)
    app, session, job, path = _prepare_import(tmp_path, bundle)
    source_path = app.state.temporary_storage.root / session.id / "source" / "signals_01.edf"
    copy_checks = 0

    def operation(report_progress, check_cancelled):
        def cancel_during_extraction():
            nonlocal copy_checks
            if source_path.exists():
                copy_checks += 1
                if copy_checks == 2:
                    app.state.job_service.request_cancel(job.id)
            check_cancelled()

        app.state.workspace_import_service.run(session, path, report_progress, cancel_during_extraction)

    execute_import_job(
        session, job.id, operation, app.state.job_executor,
        app.state.workspace_import_service, app.state.session_cleanup_service,
    )

    assert job.state == JobState.CANCELLED
    assert session.state == SessionState.FAILED
    assert not source_path.exists()
    assert not path.exists()


def test_session_deletion_cancels_import_and_removes_job_and_files(tmp_path):
    bundle, _, _ = _bundle(tmp_path)
    app, session, job, path = _prepare_import(tmp_path, bundle)
    assert app.state.session_cleanup_service.request_delete(session.id) is True
    _run_import(app, session, job, path)
    assert app.state.job_service.get(job.id) is None
    assert app.state.session_service.store.get(session.id) is None
    assert not path.exists()


def test_empty_workspace_restores_study_without_models_or_annotations(tmp_path):
    bundle, _, _ = _bundle(tmp_path)

    def change(contents):
        contents.pop("predictions/0000.json")
        contents["annotations/edits.json"] = b"[]"
        contents["provenance/models.json"] = b"[]"
        _edit_json(
            contents, "manifest.json", lambda data: data.update(tasks=[], models=[], prediction_files={}),
        )

    empty = _alter_bundle(bundle, tmp_path / "empty.zip", change)
    app, session, job, path = _prepare_import(tmp_path, empty)
    _run_import(app, session, job, path)
    assert job.state == JobState.SUCCEEDED
    assert session.state == SessionState.READY
    assert session.duration == 3600.0
    assert app.state.prediction_store.list_for_session(session.id) == []
    assert app.state.annotation_service.list_edits(session.id) == []


def test_import_preserves_original_model_provenance_for_reexport(tmp_path):
    bundle, _, _ = _bundle(tmp_path)

    def change(contents):
        _edit_json(
            contents, "provenance/models.json",
            lambda models: models[0].update(name="Original model", version="v1"),
        )
        _edit_json(
            contents, "manifest.json",
            lambda manifest: manifest["models"][0].update(name="Original model", version="v1"),
        )

    altered = _alter_bundle(bundle, tmp_path / "provenance.zip", change)
    app, session, job, path = _prepare_import(tmp_path, altered)
    _run_import(app, session, job, path)
    assert job.state == JobState.SUCCEEDED
    exporter = WorkspaceExportService(app.state.result_export_service, app.state.model_registry)
    documents, _ = exporter.build_documents(session)
    assert documents["provenance/models.json"][0]["name"] == "Original model"
    assert documents["provenance/models.json"][0]["version"] == "v1"


def test_long_import_renews_expiry_before_job_completion(tmp_path, monkeypatch):
    bundle, _, _ = _bundle(tmp_path)
    app, session, job, path = _prepare_import(tmp_path, bundle)
    session.expires_at = session.created_at - timedelta(seconds=1)
    mark_succeeded = app.state.job_service.mark_succeeded

    def complete_and_cleanup(job_id, message):
        completed = mark_succeeded(job_id, message)
        assert app.state.session_cleanup_service.cleanup_expired() == []
        return completed

    monkeypatch.setattr(app.state.job_service, "mark_succeeded", complete_and_cleanup)
    _run_import(app, session, job, path)
    assert session.state == SessionState.READY
    assert app.state.session_service.get(session.id) is session


def test_import_http_lifecycle_and_repeated_imports(tmp_path):
    bundle, original, expected = _bundle(tmp_path, PredictionKind.CONTINUOUS)
    app = create_app(temporary_storage=TemporaryStorage(tmp_path / "http-import"))
    ids = []
    edit_ids = []
    with TestClient(app) as client:
        for _ in range(2):
            with bundle.open("rb") as source:
                created = client.post(
                    "/api/v1/sessions/import", files={"bundle": (bundle.name, source, "application/zip")},
                )
            assert created.status_code == 202
            assert created.json()["state"] == "processing"
            session_id = created.json()["session_id"]
            ids.append(session_id)
            job = client.get(f"/api/v1/jobs/{created.json()['job_id']}")
            assert job.json()["state"] == "succeeded"
            session = client.get(f"/api/v1/sessions/{session_id}")
            assert session.json()["state"] == "ready"
            assert session.json()["duration"] == original.duration
            prediction = client.get(f"/api/v1/sessions/{session_id}/predictions/arousal")
            assert prediction.json()["predictions"] == expected["tasks"][0]["predictions"]
            signal = client.get(f"/api/v1/sessions/{session_id}/signals/EEG?start=0&end=1&max_points=10")
            assert signal.status_code == 200
            edit_ids.append({edit.id for edit in app.state.annotation_service.list_edits(session_id)})
            exported = client.post(
                f"/api/v1/sessions/{session_id}/exports",
                json={"format": "sleepwalker", "mode": "workspace"},
            )
            assert exported.status_code == 202
            assert client.get(f"/api/v1/exports/{exported.json()['export_id']}").json()["state"] == "ready"

    assert ids[0] != ids[1]
    assert edit_ids[0].isdisjoint(edit_ids[1])


def test_import_http_rejects_invalid_zip_and_cleans_upload(tmp_path):
    storage = TemporaryStorage(tmp_path / "uploads")
    app = create_app(temporary_storage=storage)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sessions/import", files={"bundle": ("bad.zip", b"not a zip", "application/zip")},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_BUNDLE"
    assert app.state.session_service.store.list() == []
    assert list(storage.root.iterdir()) == []


def test_import_http_rejects_unsupported_version(tmp_path):
    bundle, _, _ = _bundle(tmp_path)
    altered = _alter_bundle(
        bundle, tmp_path / "new-version.zip",
        lambda contents: _edit_json(contents, "manifest.json", lambda data: data.update(format_version=2)),
    )
    app = create_app(temporary_storage=TemporaryStorage(tmp_path / "uploads"))
    with TestClient(app) as client, altered.open("rb") as source:
        response = client.post("/api/v1/sessions/import", files={"bundle": ("new.zip", source)})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_FORMAT_VERSION"
    assert app.state.session_service.store.list() == []


def test_import_http_rejects_oversized_upload(tmp_path):
    storage = TemporaryStorage(tmp_path / "uploads", max_file_size_bytes=4)
    with TestClient(create_app(temporary_storage=storage)) as client:
        response = client.post("/api/v1/sessions/import", files={"bundle": ("large.zip", b"12345")})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"
    assert list(storage.root.iterdir()) == []
