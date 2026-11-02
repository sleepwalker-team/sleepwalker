"""Exports: a JSON result document or a re-importable workspace archive."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import RLock
from uuid import uuid4

from ..core.jobs import CancellationCheck, Job, JobExecutionError, JobPhase, JobState, ProgressCallback
from ..core.sessions import Session, utc_now
from ..core.storage import TemporaryStorage
from .annotations import AnnotationEdit, AnnotationService
from .predictions import IntervalPrediction, PredictionResult, PredictionStore, map_predictions


class ExportState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Export:
    id: str
    session_id: str
    job_id: str
    state: ExportState
    format: str
    mode: str
    created_at: datetime
    expires_at: datetime
    size: int | None = None
    path: Path | None = None


@dataclass(frozen=True)
class ResultExportOptions:
    predictions: bool = True
    probabilities: bool = True
    annotations: bool = True
    metadata: bool = True


class ExportService:
    def __init__(
        self, storage: TemporaryStorage, prediction_store: PredictionStore, annotation_service: AnnotationService,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.storage = storage
        self.prediction_store = prediction_store
        self.annotation_service = annotation_service
        self.clock = clock
        self._exports: dict[str, Export] = {}
        self._lock = RLock()

    def create(self, session: Session, job_id: str, *, format: str = "json", mode: str = "results") -> Export:
        export = Export(
            id=f"export_{uuid4()}",
            session_id=session.id,
            job_id=job_id,
            state=ExportState.QUEUED,
            format=format,
            mode=mode,
            created_at=self.clock(),
            expires_at=session.expires_at,
        )
        self.add(export)
        return export

    def add(self, export: Export) -> None:
        with self._lock:
            self._exports[export.id] = export

    def get(self, export_id: str) -> Export | None:
        with self._lock:
            return self._exports.get(export_id)

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            export_ids = [export.id for export in self._exports.values() if export.session_id == session_id]
            for export_id in export_ids:
                del self._exports[export_id]
            return len(export_ids)

    def run(self, export: Export, session: Session, options: ResultExportOptions, report_progress: ProgressCallback, check_cancelled: CancellationCheck) -> None:
        export.state = ExportState.RUNNING
        report_progress(JobPhase.EXPORTING, 0.0, "Preparing export")
        if export.mode == "workspace":
            documents, sources = self.build_workspace_documents(session)
            export.path = self.storage.write_workspace_export(session.id, export.id, documents, sources, check_cancelled)
        else:
            payload = self.build_payload(session, options)
            check_cancelled()
            export.path = self.storage.write_json_export(session.id, export.id, payload)

        export.size = export.path.stat().st_size
        report_progress(JobPhase.EXPORTING, 1.0, "Export prepared")

    def finalize(self, export_id: str, job: Job) -> Export | None:
        export = self.get(export_id)
        if export is None:
            # Session cleanup may run as soon as the job reaches a terminal state.
            return None

        if job.state == JobState.SUCCEEDED and export.path is not None:
            export.state = ExportState.READY
            return export

        self.storage.delete_export(export.session_id, export.id)
        export.path = None
        export.size = None
        export.state = ExportState.CANCELLED if job.state == JobState.CANCELLED else ExportState.FAILED
        return export

    def build_payload(self, session: Session, options: ResultExportOptions | None = None) -> dict:
        options = options or ResultExportOptions()
        payload = {"format": "sleepwalker-results", "version": 1, "exported_at": _format_datetime(self.clock())}

        if options.metadata:
            payload["session"] = _session_metadata(session)
        if options.predictions:
            payload["tasks"] = [
                _prediction_task(result, session, options.probabilities)
                for result in self.prediction_store.list_for_session(session.id)
            ]
        if options.annotations:
            payload["annotations"] = [_annotation_edit(edit) for edit in self.annotation_service.list_edits(session.id)]

        return payload

    def build_workspace_documents(self, session: Session) -> tuple[dict, dict[str, Path]]:
        sources = _source_files(session)
        payload = self.build_payload(session)
        tasks = payload["tasks"]
        models = [_model_metadata(task, session) for task in tasks]
        prediction_files = {task["task"]: f"predictions/{index:04d}.json" for index, task in enumerate(tasks)}
        documents = {
            "manifest.json": {
                "format": "sleepwalker",
                "format_version": 1,
                "created_at": payload["exported_at"],
                "created_with": _created_with(),
                "study": {"duration": session.duration, "source_files": list(sources)},
                "tasks": [task["task"] for task in tasks],
                "prediction_files": prediction_files,
                "models": models,
            },
            "study/metadata.json": payload["session"],
            "annotations/edits.json": payload["annotations"],
            "provenance/models.json": models,
        }
        for task in tasks:
            documents[prediction_files[task["task"]]] = task

        return documents, sources


def _session_metadata(session: Session) -> dict:
    return {
        "id": session.id,
        "created_at": _format_datetime(session.created_at),
        "expires_at": _format_datetime(session.expires_at),
        "format": session.format.value if session.format is not None else None,
        "duration": session.duration,
        "recording_start": _format_datetime(session.recording_start) if session.recording_start is not None else None,
        "signals": [
            {"id": signal.id, "label": signal.label, "unit": signal.unit, "sample_rate": signal.sample_rate}
            for signal in session.signals or []
        ],
    }


def _prediction_task(result: PredictionResult, session: Session, include_probabilities: bool) -> dict:
    if session.recording_start is None:
        raise ValueError("Cannot export predictions without a recording start time.")

    records = []
    for prediction in map_predictions(result, session.recording_start):
        if isinstance(prediction, IntervalPrediction):
            record = {"start": prediction.start, "end": prediction.end, "label": prediction.label}
        else:
            record = {"time": prediction.time}
            if not include_probabilities:
                record["label"] = max(prediction.probabilities, key=prediction.probabilities.__getitem__)
        if include_probabilities:
            record["probabilities"] = prediction.probabilities
        records.append(record)

    return {
        "model_id": result.model_id,
        "task": result.task,
        "kind": result.kind.value,
        "classes": list(result.classes),
        "resolution": result.resolution,
        "predictions": records,
    }


def _annotation_edit(edit: AnnotationEdit) -> dict:
    values = {
        "id": edit.id,
        "session_id": edit.session_id,
        "operation": edit.operation.value,
        "created_at": _format_datetime(edit.created_at),
        "task": edit.task,
        "start": edit.start,
        "end": edit.end,
        "label": edit.label,
        "event_id": edit.event_id,
        "reverted_edit_id": edit.reverted_edit_id,
    }
    return {key: value for key, value in values.items() if value is not None}


def _source_files(session: Session) -> dict[str, Path]:
    if not session.source_files:
        raise JobExecutionError("EXPORT_FAILED", "Study source files are missing.")

    sources = {}
    for path in session.source_files:
        name = path.name
        if not name or name in {".", ".."} or "\\" in name or ":" in name:
            raise JobExecutionError("EXPORT_FAILED", "Study source filename is invalid.")

        archive_name = f"study/source/{name}"
        if archive_name in sources:
            raise JobExecutionError("EXPORT_FAILED", "Study source filenames must be unique.")
        if not path.is_file():
            raise JobExecutionError("EXPORT_FAILED", "Study source file is missing.")

        sources[archive_name] = path

    return sources


def _model_metadata(task: dict, session: Session) -> dict:
    recorded = next((model for model in session.model_provenance if model["id"] == task["model_id"]), None)
    if recorded is not None:
        return dict(recorded)
    return {
        "id": task["model_id"],
        "task": task["task"],
        "classes": task["classes"],
        "output_kind": task["kind"],
        "output_resolution": task["resolution"],
    }


def _created_with() -> str:
    try:
        return f"sleepwalker {version('sleepwalker')}"
    except PackageNotFoundError:
        return "sleepwalker (version unknown)"


def _format_datetime(value: datetime) -> str:
    text = value.isoformat()
    return f"{text[:-6]}Z" if text.endswith("+00:00") else text
