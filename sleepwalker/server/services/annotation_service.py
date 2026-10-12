from datetime import datetime, timezone
from uuid import uuid4

from ..stores.annotation_store import AnnotationEdit, AnnotationOperation, AnnotationStore


class AnnotationEventNotFoundError(Exception):
    pass


class AnnotationEditNotFoundError(Exception):
    pass


class AnnotationService:
    def __init__(self, store: AnnotationStore) -> None:
        self.store = store

    def set_label(
        self,
        session_id: str,
        task: str,
        start: float,
        label: str,
    ) -> AnnotationEdit:
        return self._add_edit(
            session_id=session_id,
            operation=AnnotationOperation.SET_LABEL,
            task=task,
            start=start,
            label=label,
        )

    def add_event(
        self,
        session_id: str,
        task: str,
        start: float,
        end: float,
        label: str,
    ) -> AnnotationEdit:
        return self._add_edit(
            session_id=session_id,
            operation=AnnotationOperation.ADD_EVENT,
            task=task,
            start=start,
            end=end,
            label=label,
            event_id=f"evt_{uuid4()}",
        )

    def delete_event(self, session_id: str, event_id: str) -> AnnotationEdit:
        event = next(
            (
                event
                for event in self.current_events(session_id)
                if event.event_id == event_id
            ),
            None,
        )

        if event is None:
            raise AnnotationEventNotFoundError(event_id)

        return self._add_edit(
            session_id=session_id,
            operation=AnnotationOperation.DELETE_EVENT,
            task=event.task,
            event_id=event_id,
        )

    def revert(self, session_id: str, edit_id: str) -> AnnotationEdit:
        edit = self.store.get(edit_id)
        reverted_ids = self._reverted_edit_ids(session_id)

        if (
            edit is None
            or edit.session_id != session_id
            or edit.operation == AnnotationOperation.REVERT
            or edit.id in reverted_ids
        ):
            raise AnnotationEditNotFoundError(edit_id)

        return self._add_edit(
            session_id=session_id,
            operation=AnnotationOperation.REVERT,
            reverted_edit_id=edit_id,
        )

    def current_events(self, session_id: str) -> list[AnnotationEdit]:
        edits = self.store.list_for_session(session_id)
        reverted_ids = self._reverted_edit_ids(session_id)
        deleted_event_ids = {
            edit.event_id
            for edit in edits
            if edit.operation == AnnotationOperation.DELETE_EVENT
            and edit.id not in reverted_ids
        }
        return [
            edit
            for edit in edits
            if edit.operation == AnnotationOperation.ADD_EVENT
            and edit.id not in reverted_ids
            and edit.event_id not in deleted_event_ids
        ]

    def current_labels(self, session_id: str) -> list[AnnotationEdit]:
        reverted_ids = self._reverted_edit_ids(session_id)
        current: dict[tuple[str, float], AnnotationEdit] = {}

        for edit in self.store.list_for_session(session_id):
            if (
                edit.operation == AnnotationOperation.SET_LABEL
                and edit.id not in reverted_ids
                and edit.task is not None
                and edit.start is not None
            ):
                current[(edit.task, edit.start)] = edit

        return list(current.values())

    def _reverted_edit_ids(self, session_id: str) -> set[str]:
        return {
            edit.reverted_edit_id
            for edit in self.store.list_for_session(session_id)
            if edit.operation == AnnotationOperation.REVERT
            and edit.reverted_edit_id is not None
        }

    def _add_edit(
        self,
        session_id: str,
        operation: AnnotationOperation,
        **values,
    ) -> AnnotationEdit:
        edit = AnnotationEdit(
            id=f"edit_{uuid4()}",
            session_id=session_id,
            operation=operation,
            created_at=datetime.now(timezone.utc),
            **values,
        )
        self.store.add(edit)
        return edit

    def delete_session(self, session_id: str) -> int:
        return self.store.delete_session(session_id)
