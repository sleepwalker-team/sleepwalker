import math
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ..storage.temporary_storage import TemporaryStorage
from ..stores.annotation_store import AnnotationEdit, AnnotationOperation, AnnotationStore
from ..stores.prediction_store import PredictionKind, PredictionResult, PredictionStore
from ..stores.session_store import Session, SessionState
from ..stores.job_store import Job, JobPhase, JobState
from ..validation import InvalidStudyError, UnsupportedStudyFormatError, validate_study
from ..validation.workspace import (
    InvalidWorkspaceError,
    LoadedWorkspace,
    UnsupportedWorkspaceVersionError,
    WorkspaceInterval,
    WorkspaceLimits,
    WorkspaceSample,
    WorkspaceTask,
    inspect_workspace,
    load_workspace,
)
from .annotation_service import AnnotationService
from .job_executor import CancellationCheck, JobExecutionError, ProgressCallback
from .prediction_mapper import validate_prediction_result
from .session_service import SessionService


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
            metadata = workspace.metadata
            if (
                study.format != metadata.format
                or not math.isclose(study.duration, metadata.duration)
                or not math.isclose(study.duration, workspace.manifest.study.duration)
                or study.recording_start != metadata.recording_start
                or [vars(signal) for signal in study.signals or []]
                != [signal.model_dump() for signal in metadata.signals]
            ):
                raise InvalidWorkspaceError("Workspace metadata does not match the source recording.")

            results = [_restore_prediction(task, session.id, check_cancelled) for task in workspace.tasks]
            _validate_task_metadata(workspace)
            for result in results:
                check_cancelled()
                validate_prediction_result(result, study.recording_start)

            edits = _restore_edits(workspace, session.id, check_cancelled)
            check_cancelled()
            report_progress(JobPhase.IMPORTING, 0.8, "Restoring workspace data")
            self.prediction_store.add_many(results)
            for edit in edits:
                self.annotation_service.store.add(edit)

            session.source_files = study.source_files
            session.format = study.format
            session.duration = study.duration
            session.recording_start = study.recording_start
            session.signals = study.signals
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
        except (InvalidWorkspaceError, InvalidStudyError, UnsupportedStudyFormatError, ValueError) as exc:
            raise JobExecutionError("INVALID_BUNDLE", "Workspace data is invalid or inconsistent.") from exc

    def finalize(self, session: Session, job: Job) -> None:
        if self.session_service.store.get(session.id) is None:
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
            session.model_provenance = None
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
            if interval.end <= interval.start:
                raise InvalidWorkspaceError("Workspace prediction interval is invalid.")
            if task.kind == PredictionKind.EPOCH:
                if not math.isclose(interval.end - interval.start, task.resolution):
                    raise InvalidWorkspaceError("Workspace epoch width does not match its resolution.")
                row = {"time": (interval.start + interval.end) / 2}
            else:
                row = {"start": interval.start, "end": interval.end}
        if set(probabilities) != set(task.classes):
            raise InvalidWorkspaceError("Workspace prediction classes do not match its probabilities.")
        if task.kind != PredictionKind.CONTINUOUS:
            if interval.label != max(task.classes, key=probabilities.__getitem__):
                raise InvalidWorkspaceError("Workspace prediction label does not match its probabilities.")
        if "time" in row:
            if row["time"] in times:
                raise InvalidWorkspaceError("Workspace contains duplicate prediction times.")
            times.add(row["time"])
        row.update({f"prob__{label}": probability for label, probability in probabilities.items()})
        rows.append(row)

    columns = ["start", "end"] if task.kind == PredictionKind.EVENT else ["time"]
    columns += [f"prob__{label}" for label in task.classes]
    return PredictionResult(
        session_id=session_id,
        model_id=task.model_id,
        task=task.task,
        classes=tuple(task.classes),
        resolution=task.resolution,
        kind=task.kind,
        frame=pd.DataFrame(rows, columns=columns),
    )


def _validate_task_metadata(workspace: LoadedWorkspace) -> None:
    tasks = workspace.tasks
    if [task.task for task in tasks] != workspace.manifest.tasks:
        raise InvalidWorkspaceError("Workspace task files do not match its manifest.")
    models = {model.id: model for model in workspace.manifest.models}
    if len(models) != len(workspace.manifest.models) or set(models) != {task.model_id for task in tasks}:
        raise InvalidWorkspaceError("Workspace model provenance is invalid.")
    for task in tasks:
        model = models[task.model_id]
        if (
            model.task != task.task or model.classes != task.classes
            or model.output_kind != task.kind or model.output_resolution != task.resolution
        ):
            raise InvalidWorkspaceError("Workspace model provenance does not match its results.")


def _restore_edits(
    workspace: LoadedWorkspace,
    session_id: str,
    check_cancelled: CancellationCheck,
) -> list[AnnotationEdit]:
    tasks = {task.task: task for task in workspace.tasks}
    old_edits = {}
    event_ids = {}
    edit_ids = {}
    temporary_service = AnnotationService(AnnotationStore())
    restored = []
    for edit in workspace.edits:
        check_cancelled()
        if (
            edit.id in old_edits or edit.session_id != workspace.metadata.id
            or edit.created_at.tzinfo is None
        ):
            raise InvalidWorkspaceError("Workspace annotation identity is invalid.")
        values = edit.model_dump()
        values["id"] = f"edit_{uuid4()}"
        values["session_id"] = session_id
        operation = edit.operation
        task = tasks.get(edit.task)
        if operation in {AnnotationOperation.SET_LABEL, AnnotationOperation.ADD_EVENT}:
            if task is None or edit.start is None or edit.label not in task.classes:
                raise InvalidWorkspaceError("Workspace annotation refers to an invalid task or label.")
            if edit.start < 0 or edit.start >= workspace.metadata.duration or edit.reverted_edit_id is not None:
                raise InvalidWorkspaceError("Workspace annotation range is invalid.")
            if operation == AnnotationOperation.SET_LABEL:
                if task.kind != PredictionKind.EPOCH or edit.end is not None or edit.event_id is not None:
                    raise InvalidWorkspaceError("Workspace label edit is invalid.")
                aligned = math.isclose(
                    edit.start / task.resolution, round(edit.start / task.resolution),
                    rel_tol=0.0, abs_tol=1e-9,
                )
                if edit.start + task.resolution > workspace.metadata.duration or not aligned:
                    raise InvalidWorkspaceError("Workspace label edit is not aligned to an epoch.")
            else:
                if (
                    task.kind == PredictionKind.EPOCH or edit.end is None
                    or not edit.start < edit.end <= workspace.metadata.duration
                    or edit.event_id is None or edit.event_id in event_ids
                ):
                    raise InvalidWorkspaceError("Workspace event edit is invalid.")
                event_ids[edit.event_id] = f"evt_{uuid4()}"
                values["event_id"] = event_ids[edit.event_id]
        elif operation == AnnotationOperation.DELETE_EVENT:
            if edit.event_id not in event_ids or any(
                value is not None for value in (edit.start, edit.end, edit.label, edit.reverted_edit_id)
            ):
                raise InvalidWorkspaceError("Workspace event deletion is invalid.")
            event_id = event_ids[edit.event_id]
            active = {event.event_id: event for event in temporary_service.current_events(session_id)}
            if event_id not in active or active[event_id].task != edit.task:
                raise InvalidWorkspaceError("Workspace event deletion refers to an inactive event.")
            values["event_id"] = event_id
        else:
            if edit.reverted_edit_id not in old_edits or any(
                value is not None for value in (edit.task, edit.start, edit.end, edit.label, edit.event_id)
            ):
                raise InvalidWorkspaceError("Workspace annotation revert is invalid.")
            target = old_edits[edit.reverted_edit_id]
            previous_reverts = {previous.reverted_edit_id for previous in old_edits.values()}
            if target.operation == AnnotationOperation.REVERT or target.id in previous_reverts:
                raise InvalidWorkspaceError("Workspace annotation revert refers to an invalid edit.")
            values["reverted_edit_id"] = edit_ids[edit.reverted_edit_id]

        restored_edit = AnnotationEdit(**values)
        temporary_service.store.add(restored_edit)
        restored.append(restored_edit)
        old_edits[edit.id] = edit
        edit_ids[edit.id] = restored_edit.id

    return restored
