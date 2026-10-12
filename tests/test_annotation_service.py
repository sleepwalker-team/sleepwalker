import pytest

from sleepwalker.server.services.annotation_service import (
    AnnotationEditNotFoundError,
    AnnotationEventNotFoundError,
    AnnotationService,
)
from sleepwalker.server.stores.annotation_store import AnnotationOperation, AnnotationStore


def test_set_label_keeps_latest_active_edit_and_can_revert_it() -> None:
    service = AnnotationService(AnnotationStore())
    first = service.set_label("sess-test", "sleep_stage", 30.0, "n2")
    second = service.set_label("sess-test", "sleep_stage", 30.0, "n3")

    assert service.current_labels("sess-test") == [second]

    revert = service.revert("sess-test", second.id)

    assert revert.operation == AnnotationOperation.REVERT
    assert revert.reverted_edit_id == second.id
    assert service.current_labels("sess-test") == [first]


def test_event_delete_and_revert_preserve_history() -> None:
    store = AnnotationStore()
    service = AnnotationService(store)
    event = service.add_event("sess-test", "arousal", 12.0, 18.0, "arousal")

    assert service.current_events("sess-test") == [event]

    deletion = service.delete_event("sess-test", event.event_id or "")

    assert service.current_events("sess-test") == []
    assert len(store.list_for_session("sess-test")) == 2

    service.revert("sess-test", deletion.id)

    assert service.current_events("sess-test") == [event]
    assert len(store.list_for_session("sess-test")) == 3


def test_annotation_service_rejects_missing_event_and_edit() -> None:
    service = AnnotationService(AnnotationStore())

    with pytest.raises(AnnotationEventNotFoundError):
        service.delete_event("sess-test", "evt-unknown")

    with pytest.raises(AnnotationEditNotFoundError):
        service.revert("sess-test", "edit-unknown")


def test_annotation_store_deletes_all_session_edits() -> None:
    store = AnnotationStore()
    service = AnnotationService(store)
    service.set_label("sess-a", "sleep_stage", 0.0, "wake")
    service.set_label("sess-b", "sleep_stage", 0.0, "n1")

    assert store.delete_session("sess-a") == 1
    assert store.list_for_session("sess-a") == []
    assert len(store.list_for_session("sess-b")) == 1

def test_annotation_service_deletes_session_edits() -> None:
    service = AnnotationService(AnnotationStore())

    session_edit = service.set_label(
        session_id="sess-test",
        task="sleep_stage",
        start=0.0,
        label="wake",
    )
    other_edit = service.set_label(
        session_id="sess-other",
        task="sleep_stage",
        start=0.0,
        label="n1",
    )

    assert service.delete_session("sess-test") == 1
    assert service.store.get(session_edit.id) is None
    assert service.store.get(other_edit.id) is other_edit