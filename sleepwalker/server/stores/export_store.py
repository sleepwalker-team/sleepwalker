from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import RLock


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


class ExportStore:
    def __init__(self) -> None:
        self._exports: dict[str, Export] = {}
        self._lock = RLock()

    def add(self, export: Export) -> None:
        with self._lock:
            if export.id in self._exports:
                raise ValueError(f"Export ID is already registered: {export.id}")

            self._exports[export.id] = export

    def get(self, export_id: str) -> Export | None:
        with self._lock:
            return self._exports.get(export_id)

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            export_ids = [
                export.id
                for export in self._exports.values()
                if export.session_id == session_id
            ]

            for export_id in export_ids:
                del self._exports[export_id]

            return len(export_ids)
