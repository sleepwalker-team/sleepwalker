import pytest

from sleepwalker.server.results.annotations import AnnotationEditNotFoundError, AnnotationEventNotFoundError, AnnotationService
from sleepwalker.server.results.annotations import AnnotationOperation


def test_set_label_keeps_latest_active_edit_and_can_revert_it() -> None:
    service = AnnotationService()
    first = service.set_label("sess-test", "sleep_stage", 30.0, "n2")
    second = service.set_label("sess-test", "sleep_stage", 30.0, "n3")

    assert service.snapshot("sess-test").labels == [second]

    revert = service.revert("sess-test", second.id)

    assert revert.operation == AnnotationOperation.REVERT
    assert revert.reverted_edit_id == second.id
    assert service.snapshot("sess-test").labels == [first]


def test_event_delete_and_revert_preserve_history() -> None:
    store = service = AnnotationService()
    event = service.add_event("sess-test", "arousal", 12.0, 18.0, "arousal")

    assert service.snapshot("sess-test").events == [event]

    deletion = service.delete_event("sess-test", event.event_id or "")

    assert service.snapshot("sess-test").events == []
    assert len(store.list_edits("sess-test")) == 2

    service.revert("sess-test", deletion.id)

    assert service.snapshot("sess-test").events == [event]
    assert len(store.list_edits("sess-test")) == 3


def test_annotation_service_rejects_missing_event_and_edit() -> None:
    service = AnnotationService()

    with pytest.raises(AnnotationEventNotFoundError):
        service.delete_event("sess-test", "evt-unknown")

    with pytest.raises(AnnotationEditNotFoundError):
        service.revert("sess-test", "edit-unknown")


def test_annotation_store_deletes_all_session_edits() -> None:
    store = service = AnnotationService()
    service.set_label("sess-a", "sleep_stage", 0.0, "wake")
    service.set_label("sess-b", "sleep_stage", 0.0, "n1")

    assert store.delete_session("sess-a") == 1
    assert store.list_edits("sess-a") == []
    assert len(store.list_edits("sess-b")) == 1

def test_annotation_service_deletes_session_edits() -> None:
    service = AnnotationService()

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
    assert service.list_edits("sess-test") == []
    assert service.list_edits(other_edit.session_id) == [other_edit]


def test_annotation_snapshot_keeps_history_and_resolved_state_consistent() -> None:
    service = AnnotationService()
    label = service.set_label("sess-test", "sleep_stage", 30.0, "n2")
    event = service.add_event("sess-test", "arousal", 12.0, 18.0, "arousal")
    snapshot = service.snapshot("sess-test")
    service.set_label("sess-test", "sleep_stage", 30.0, "n3")
    service.delete_event("sess-test", event.event_id)

    assert snapshot.edits == [label, event]
    assert snapshot.labels == [label]
    assert snapshot.events == [event]
    current = service.snapshot("sess-test")
    assert len(current.edits) == 4
    assert current.labels[0].label == "n3"
    assert current.events == []
