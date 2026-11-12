"""Sleepwalker workspace archives: format, safety validation and restoration."""

import hashlib
import json
import math
import re
import stat
import zlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from zipfile import BadZipFile, ZipFile

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..core.jobs import CancellationCheck, Job, JobExecutionError, JobPhase, JobState, ProgressCallback
from ..core.sessions import Session, SessionService, SessionState
from ..core.storage import DEFAULT_MAX_FILE_SIZE_BYTES, TemporaryStorage
from ..recording.study import InvalidStudyError, StudyInfo, UnsupportedStudyFormatError, validate_study
from .annotations import AnnotationEdit, AnnotationEditNotFoundError, AnnotationOperation, AnnotationService
from .predictions import PredictionKind, PredictionResult, PredictionStore, map_predictions, model_events

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveNumber = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
Text = Annotated[str, Field(strict=True, min_length=1)]


class InvalidWorkspaceError(Exception):
    pass


class UnsupportedWorkspaceVersionError(Exception):
    pass


@dataclass(frozen=True)
class WorkspaceLimits:
    max_total_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
    max_document_bytes: int = 256 * 1024**2
    max_manifest_bytes: int = 1024**2
    max_members: int = 10000


class WorkspaceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspaceModel(WorkspaceDocument):
    id: Text
    task: Text
    classes: list[Text] = Field(min_length=1)
    output_kind: PredictionKind
    output_resolution: PositiveNumber
    name: Text | None = None
    version: Text | None = None
    input_channels: list[Text] | None = None
    channel_map: dict[Text, Text] | None = None


class WorkspaceManifest(WorkspaceDocument):
    format: Literal["sleepwalker"]
    format_version: Literal[2]
    created_at: datetime
    created_with: Text
    source_files: list[Text] = Field(min_length=1)
    prediction_files: dict[Text, Text]
    models: list[WorkspaceModel]
    checksums: dict[Text, Text]


class WorkspaceTask(WorkspaceDocument):
    model_id: Text
    task: Text
    kind: PredictionKind
    classes: list[Text] = Field(min_length=1)
    resolution: PositiveNumber
    predictions: list[dict]


class WorkspaceInterval(WorkspaceDocument):
    start: Number
    end: Number
    label: Text
    probabilities: dict[Text, Number]


class WorkspaceSample(WorkspaceDocument):
    time: Number
    probabilities: dict[Text, Number]


class WorkspaceEdit(WorkspaceDocument):
    id: Text
    session_id: Text
    operation: AnnotationOperation
    created_at: datetime
    task: Text | None = None
    start: Number | None = None
    end: Number | None = None
    label: Text | None = None
    event_id: Text | None = None
    reverted_edit_id: Text | None = None


@dataclass
class LoadedWorkspace:
    manifest: WorkspaceManifest
    tasks: list[WorkspaceTask]
    edits: list[WorkspaceEdit]
    sources: list[Path]


def inspect_workspace(path: Path, limits: WorkspaceLimits) -> WorkspaceManifest:
    try:
        with ZipFile(path) as archive:
            return _inspect_archive(archive, limits)
    except (
        BadZipFile, UnicodeError, ValueError, KeyError,
        NotImplementedError, RecursionError, zlib.error,
    ) as exc:
        raise InvalidWorkspaceError("Workspace archive or manifest is invalid.") from exc


def _inspect_archive(archive: ZipFile, limits: WorkspaceLimits) -> WorkspaceManifest:
    members = archive.infolist()
    if not members or len(members) > limits.max_members:
        raise InvalidWorkspaceError("Workspace archive contains too many or no files.")

    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise InvalidWorkspaceError("Workspace archive contains duplicate filenames.")

    if sum(member.file_size for member in members) > limits.max_total_bytes:
        raise InvalidWorkspaceError("Workspace exceeds the uncompressed size limit.")

    for member in members:
        name = member.filename
        mode = member.external_attr >> 16
        if (
            not name or "\\" in name or "\x00" in member.orig_filename or ":" in name
            or PurePosixPath(name).is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or member.is_dir() or member.flag_bits & 1
            or (stat.S_IFMT(mode) not in {0, stat.S_IFREG})
        ):
            raise InvalidWorkspaceError("Workspace archive contains an unsafe file entry.")

    if "manifest.json" not in names:
        raise InvalidWorkspaceError("Workspace manifest is missing.")
    if archive.getinfo("manifest.json").file_size > limits.max_manifest_bytes:
        raise InvalidWorkspaceError("Workspace manifest exceeds the size limit.")

    raw = _read_json(archive.read("manifest.json"))
    if not isinstance(raw, dict) or raw.get("format") != "sleepwalker":
        raise InvalidWorkspaceError("Unsupported workspace format.")
    version = raw.get("format_version")
    if type(version) is not int:
        raise InvalidWorkspaceError("Workspace format version is invalid.")
    if version != 2:
        raise UnsupportedWorkspaceVersionError("Workspace format version is not supported.")

    manifest = WorkspaceManifest.model_validate(raw)
    sources = manifest.source_files
    if len(sources) != len(set(sources)) or any(
        not name.startswith("study/source/") or len(name.split("/")) != 3
        for name in sources
    ):
        raise InvalidWorkspaceError("Workspace source file mapping is invalid.")

    predictions = list(manifest.prediction_files.values())
    if len(predictions) != len(set(predictions)) or any(
        not name.startswith("predictions/") or not name.endswith(".json")
        or len(name.split("/")) != 2 for name in predictions
    ):
        raise InvalidWorkspaceError("Workspace prediction file mapping is invalid.")

    expected = {"annotations/edits.json", *sources, *predictions}
    if set(names) != expected | {"manifest.json"} or set(manifest.checksums) != expected:
        raise InvalidWorkspaceError("Workspace required files or checksums do not match.")
    if any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in manifest.checksums.values()):
        raise InvalidWorkspaceError("Workspace checksum is invalid.")

    for name in expected - set(sources):
        if archive.getinfo(name).file_size > limits.max_document_bytes:
            raise InvalidWorkspaceError("Workspace JSON document exceeds the size limit.")

    return manifest


def load_workspace(
    path: Path,
    source_directory: Path,
    limits: WorkspaceLimits,
    check_cancelled: Callable[[], None],
) -> LoadedWorkspace:
    try:
        with ZipFile(path) as archive:
            manifest = _inspect_archive(archive, limits)
            source_directory.mkdir(parents=True, exist_ok=False)
            sources = []
            documents = {}
            for name, expected in manifest.checksums.items():
                check_cancelled()
                checksum = hashlib.sha256()
                data = bytearray()
                source_path = source_directory / PurePosixPath(name).name
                is_source = name in manifest.source_files
                with archive.open(name) as entry:
                    if is_source:
                        with source_path.open("xb") as target:
                            while chunk := entry.read(1024 * 1024):
                                check_cancelled()
                                target.write(chunk)
                                checksum.update(chunk)
                    else:
                        while chunk := entry.read(1024 * 1024):
                            check_cancelled()
                            data.extend(chunk)
                            checksum.update(chunk)

                if checksum.hexdigest() != expected:
                    raise InvalidWorkspaceError("Workspace checksum verification failed.")
                if is_source:
                    sources.append(source_path)
                else:
                    documents[name] = _read_json(data)

            tasks = [WorkspaceTask.model_validate(documents[name]) for name in manifest.prediction_files.values()]
            raw_edits = documents["annotations/edits.json"]
            if not isinstance(raw_edits, list):
                raise InvalidWorkspaceError("Workspace annotation history must be a list.")
            edits = [WorkspaceEdit.model_validate(edit) for edit in raw_edits]

            return LoadedWorkspace(manifest, tasks, edits, sources)
    except (
        BadZipFile, UnicodeError, ValueError, KeyError,
        NotImplementedError, RecursionError, zlib.error,
    ) as exc:
        raise InvalidWorkspaceError("Workspace archive data is invalid.") from exc


def _read_json(data: bytes | bytearray):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key.")
            result[key] = value
        return result

    return json.loads(data.decode("utf-8"), object_pairs_hook=unique_keys)


class WorkspaceImportService:
    def __init__(self, session_service: SessionService, storage: TemporaryStorage, prediction_store: PredictionStore, annotation_service: AnnotationService, limits: WorkspaceLimits | None = None ) -> None:
        self.session_service = session_service
        self.storage = storage
        self.prediction_store = prediction_store
        self.annotation_service = annotation_service
        self.limits = limits or WorkspaceLimits(max_total_bytes=storage.max_file_size_bytes)

    def inspect(self, path: Path) -> None:
        inspect_workspace(path, self.limits)

    def run(
        self,
        session: Session,
        bundle_path: Path,
        report_progress: ProgressCallback,
        check_cancelled: CancellationCheck,
    ) -> None:
        report_progress(JobPhase.IMPORTING, 0.0, "Validating workspace")
        try:
            workspace = load_workspace(
                bundle_path, self.storage.root / session.id / "source", self.limits, check_cancelled,
            )
            check_cancelled()
            study = validate_study(workspace.sources)
            results = [_restore_prediction(task, session.id, check_cancelled) for task in workspace.tasks]
            for result in results:
                check_cancelled()
                map_predictions(result, study.recording_start)

            edits = _restore_edits(workspace, results, session.id, study, check_cancelled)
            check_cancelled()
            report_progress(JobPhase.IMPORTING, 0.8, "Restoring workspace data")
            self.prediction_store.add_many(results)
            for edit in edits:
                self.annotation_service.add(edit)

            session.apply_study(study)
            session.model_provenance = [
                model.model_dump(mode="json", exclude_none=True) for model in workspace.manifest.models
            ]
            bundle_path.unlink()
            # Renew before the job becomes terminal, so expiry cleanup cannot
            # discard an import that took longer than the original session TTL.
            now = self.session_service.clock()
            session.last_accessed_at = now
            session.expires_at = now + self.session_service.ttl
            report_progress(JobPhase.IMPORTING, 1.0, "Workspace restored")
        except UnsupportedWorkspaceVersionError as exc:
            raise JobExecutionError("UNSUPPORTED_FORMAT_VERSION", str(exc)) from exc
        except (InvalidWorkspaceError, InvalidStudyError, UnsupportedStudyFormatError, ValueError, LookupError, AnnotationEditNotFoundError) as exc:
            raise JobExecutionError("INVALID_BUNDLE", "Workspace data is invalid or inconsistent.") from exc

    def finalize(self, session: Session, job: Job) -> None:
        if self.session_service.peek(session.id) is None:
            return
        if job.state == JobState.SUCCEEDED:
            session.state = SessionState.READY
        else:
            self.storage.delete_session(session.id)
            self.prediction_store.delete_session(session.id)
            self.annotation_service.delete_session(session.id)
            session.source_files = []
            session.format = None
            session.duration = None
            session.recording_start = None
            session.signals = None
            session.patient = {}
            session.model_provenance = []
            session.state = SessionState.FAILED


def _restore_prediction(
    task: WorkspaceTask,
    session_id: str,
    check_cancelled: CancellationCheck,
) -> PredictionResult:
    if len(task.classes) != len(set(task.classes)):
        raise InvalidWorkspaceError("Workspace task contains duplicate classes.")
    rows = []
    times = set()
    for raw in task.predictions:
        check_cancelled()
        if task.kind == PredictionKind.CONTINUOUS:
            sample = WorkspaceSample.model_validate(raw)
            row = {"time": sample.time}
            probabilities = sample.probabilities
        else:
            interval = WorkspaceInterval.model_validate(raw)
            probabilities = interval.probabilities
            if not math.isclose(interval.end - interval.start, task.resolution):
                raise InvalidWorkspaceError("Workspace interval width does not match its resolution.")
            row = {"time": (interval.start + interval.end) / 2}
        if set(probabilities) != set(task.classes):
            raise InvalidWorkspaceError("Workspace prediction classes do not match its probabilities.")
        if task.kind != PredictionKind.CONTINUOUS:
            if interval.label != max(task.classes, key=probabilities.__getitem__):
                raise InvalidWorkspaceError("Workspace prediction label does not match its probabilities.")
        if row["time"] in times:
            raise InvalidWorkspaceError("Workspace contains duplicate prediction times.")
        times.add(row["time"])
        row.update({f"prob__{label}": probability for label, probability in probabilities.items()})
        rows.append(row)

    columns = ["time", *(f"prob__{label}" for label in task.classes)]
    return PredictionResult(
        session_id=session_id,
        model_id=task.model_id,
        task=task.task,
        classes=tuple(task.classes),
        resolution=task.resolution,
        kind=task.kind,
        frame=pd.DataFrame(rows, columns=columns),
    )


def _restore_edits(
    workspace: LoadedWorkspace, results: list[PredictionResult], session_id: str, study: StudyInfo, check_cancelled: CancellationCheck,
) -> list[AnnotationEdit]:
    """Replay the edit history through the same rules as the API; IDs are new, timestamps are kept."""
    if len({edit.id for edit in workspace.edits}) != len(workspace.edits):
        raise InvalidWorkspaceError("Workspace contains duplicate annotation edits.")
    tasks = {result.task: result for result in results}
    model_event_ids = {result.task: {event.id for event in model_events(result, study.recording_start)} for result in results}
    replay = AnnotationService()
    edit_ids, event_ids, created = {}, {}, {}
    for edit in workspace.edits:
        check_cancelled()
        if edit.operation == AnnotationOperation.REVERT:
            restored = replay.revert(session_id, edit_ids.get(edit.reverted_edit_id))
        else:
            task = tasks.get(edit.task)
            if task is None:
                raise InvalidWorkspaceError("Workspace annotation refers to an unknown task.")
            restored = replay.record(
                session_id, edit.operation, task, duration=study.duration, recording_start=study.recording_start,
                model_event_ids=model_event_ids[task.task], start=edit.start, end=edit.end, label=edit.label,
                event_id=event_ids.get(edit.event_id, edit.event_id),
            )
            if edit.operation == AnnotationOperation.ADD_EVENT:
                event_ids[edit.event_id] = restored.event_id
        edit_ids[edit.id] = restored.id
        created[restored.id] = edit.created_at
    return [replace(edit, created_at=created[edit.id]) for edit in replay.list_edits(session_id)]
