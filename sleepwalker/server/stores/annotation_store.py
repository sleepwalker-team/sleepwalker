from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from threading import RLock


class AnnotationOperation(str, Enum):
    SET_LABEL = "set_label"
    ADD_EVENT = "add_event"
    DELETE_EVENT = "delete_event"
    REVERT = "revert"


@dataclass(frozen=True)
class AnnotationEdit:
    id: str
    session_id: str
    operation: AnnotationOperation
    created_at: datetime
    task: str | None = None
    start: float | None = None
    end: float | None = None
    label: str | None = None
    event_id: str | None = None
    reverted_edit_id: str | None = None


class AnnotationStore:
    def __init__(self) -> None:
        self._edits: dict[str, AnnotationEdit] = {}
        self._lock = RLock()

    def add(self, edit: AnnotationEdit) -> None:
        with self._lock:
            if edit.id in self._edits:
                raise ValueError(f"Annotation edit ID is already registered: {edit.id}")

            self._edits[edit.id] = edit

    def get(self, edit_id: str) -> AnnotationEdit | None:
        with self._lock:
            return self._edits.get(edit_id)

    def list_for_session(self, session_id: str) -> list[AnnotationEdit]:
        with self._lock:
            return [edit for edit in self._edits.values() if edit.session_id == session_id]

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            edit_ids = [edit.id for edit in self._edits.values() if edit.session_id == session_id]

            for edit_id in edit_ids:
                del self._edits[edit_id]

            return len(edit_ids)
