"""Human corrections, stored as an append-only list of edits per session."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from operator import attrgetter
from threading import RLock
from uuid import uuid4

from .predictions import IntervalPrediction, ModelEvent, PredictionKind, PredictionResult


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


class AnnotationEditNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class AnnotationSnapshot:
    edits: list[AnnotationEdit]
    labels: list[AnnotationEdit]
    events: list[AnnotationEdit]
    deleted_events: list[AnnotationEdit]


class AnnotationService:
    def __init__(self) -> None:
        self._edits: dict[str, AnnotationEdit] = {}
        self._lock = RLock()

    def set_label(self, session_id: str, task: str, start: float, label: str) -> AnnotationEdit:
        return self._add_edit(session_id, AnnotationOperation.SET_LABEL, task=task, start=start, label=label)

    def add_event(self, session_id: str, task: str, start: float, end: float, label: str) -> AnnotationEdit:
        return self._add_edit(session_id, AnnotationOperation.ADD_EVENT, task=task, start=start, end=end, label=label, event_id=f"evt_{uuid4()}")

    def delete_event(self, session_id: str, task: str, event_id: str) -> AnnotationEdit:
        """Delete a human or model event; callers check that it exists and is still active."""
        return self._add_edit(session_id, AnnotationOperation.DELETE_EVENT, task=task, event_id=event_id)

    def revert(self, session_id: str, edit_id: str) -> AnnotationEdit:
        with self._lock:
            edit = self._edits.get(edit_id)
        if (
            edit is None
            or edit.session_id != session_id
            or edit.id in self._reverted_ids(self.list_edits(session_id))
        ):
            raise AnnotationEditNotFoundError(edit_id)
        return self._add_edit(session_id, AnnotationOperation.REVERT, reverted_edit_id=edit_id)

    def snapshot(self, session_id: str) -> AnnotationSnapshot:
        edits = self.list_edits(session_id)
        reverted = self._reverted_ids(edits)
        active = [edit for edit in edits if edit.id not in reverted]
        deletions = [edit for edit in active if edit.operation == AnnotationOperation.DELETE_EVENT]
        deleted_ids = {edit.event_id for edit in deletions}
        events = [edit for edit in active if edit.operation == AnnotationOperation.ADD_EVENT and edit.event_id not in deleted_ids]
        # A later label for the same epoch replaces the earlier one.
        labels = {(edit.task, edit.start): edit for edit in active if edit.operation == AnnotationOperation.SET_LABEL}
        return AnnotationSnapshot(edits=edits, labels=list(labels.values()), events=events, deleted_events=deletions)

    def list_edits(self, session_id: str) -> list[AnnotationEdit]:
        with self._lock:
            return [edit for edit in self._edits.values() if edit.session_id == session_id]

    def add(self, edit: AnnotationEdit) -> None:
        with self._lock:
            self._edits[edit.id] = edit

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            edit_ids = [edit.id for edit in self._edits.values() if edit.session_id == session_id]
            for edit_id in edit_ids:
                del self._edits[edit_id]
            return len(edit_ids)

    @staticmethod
    def _reverted_ids(edits: list[AnnotationEdit]) -> set[str]:
        """Targets of active reverts. Reverting a revert restores its target (redo).

        A revert always targets an earlier edit, so one pass from the end decides which reverts are active.
        """
        reverted = set()
        for edit in reversed(edits):
            if edit.operation == AnnotationOperation.REVERT and edit.id not in reverted:
                reverted.add(edit.reverted_edit_id)
        return reverted

    def _add_edit(self, session_id: str, operation: AnnotationOperation, **values) -> AnnotationEdit:
        edit = AnnotationEdit(id=f"edit_{uuid4()}", session_id=session_id, operation=operation, created_at=datetime.now(timezone.utc), **values)
        self.add(edit)
        return edit


@dataclass(frozen=True)
class EffectiveAnnotation:
    """One interval of model output with the active human edits applied."""

    start: float
    end: float
    label: str
    source: str  # "model" or "human"
    confidence: float | None = None
    id: str | None = None


def effective_annotations(
    result: PredictionResult, records: list[IntervalPrediction], events: list[ModelEvent], snapshot: AnnotationSnapshot,
) -> list[EffectiveAnnotation]:
    if result.kind == PredictionKind.EPOCH:
        # Rounded to the 1 µs tolerance with which set_label matched the epoch.
        labels = {round(edit.start, 6): edit.label for edit in snapshot.labels if edit.task == result.task}
        return [
            EffectiveAnnotation(r.start, r.end, labels[round(r.start, 6)], "human") if round(r.start, 6) in labels
            else EffectiveAnnotation(r.start, r.end, r.label, "model", r.probabilities[r.label])
            for r in records
        ]
    deleted = {edit.event_id for edit in snapshot.deleted_events}
    model = [EffectiveAnnotation(e.start, e.end, e.label, "model", e.confidence, e.id) for e in events if e.id not in deleted]
    human = [EffectiveAnnotation(e.start, e.end, e.label, "human", id=e.event_id) for e in snapshot.events if e.task == result.task]
    return sorted(model + human, key=attrgetter("start"))
