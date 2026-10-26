from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..stores.session_store import Session
from .job_executor import JobExecutionError
from .model_registry import ModelRegistry
from .result_export_service import ResultExportService


class WorkspaceExportService:
    def __init__(self, result_export_service: ResultExportService, model_registry: ModelRegistry) -> None:
        self.result_export_service = result_export_service
        self.model_registry = model_registry

    def build_documents(self, session: Session) -> tuple[dict, dict[str, Path]]:
        sources = self._source_files(session)
        payload = self.result_export_service.build_payload(session)
        tasks = payload["tasks"]
        models = [self._model_metadata(task, session) for task in tasks]
        prediction_files = {
            task["task"]: f"predictions/{index:04d}.json"
            for index, task in enumerate(tasks)
        }
        documents = {
            "manifest.json": {
                "format": "sleepwalker",
                "format_version": 1,
                "created_at": payload["exported_at"],
                "created_with": _created_with(),
                "study": {
                    "duration": session.duration,
                    "source_files": list(sources),
                },
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

    @staticmethod
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

    def _model_metadata(self, task: dict, session: Session) -> dict:
        imported = next(
            (model for model in session.model_provenance or [] if model["id"] == task["model_id"]),
            None,
        )
        if imported is not None:
            return dict(imported)

        metadata = {
            "id": task["model_id"],
            "task": task["task"],
            "classes": task["classes"],
            "output_kind": task["kind"],
            "output_resolution": task["resolution"],
        }
        model = self.model_registry.get(task["model_id"])
        if model is not None:
            metadata.update({
                "name": model.name,
                "input_channels": list(model.input_channels),
                "capabilities": list(model.capabilities),
            })

        return metadata


def _created_with() -> str:
    try:
        return f"sleepwalker {version('sleepwalker')}"
    except PackageNotFoundError:
        return "sleepwalker (version unknown)"
