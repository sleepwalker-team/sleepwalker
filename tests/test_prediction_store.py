import pandas as pd
import pytest

from sleepwalker.server.results.predictions import PredictionResult, PredictionStore


def create_result(
    session_id: str = "sess-a",
    model_id: str = "model-a",
    task: str = "sleep_stage",
) -> PredictionResult:
    return PredictionResult(
        session_id=session_id,
        model_id=model_id,
        task=task,
        classes=("wake", "n1"),
        resolution=30.0,
        frame=pd.DataFrame({"prediction": ["wake"]}),
    )


def test_prediction_store_adds_lists_and_deletes_results() -> None:
    store = PredictionStore()
    result_b = create_result(model_id="model-b", task="arousal")
    result_a = create_result(model_id="model-a")

    store.add_many([result_b, result_a])

    assert store.get_by_task("sess-a", "sleep_stage") is not None
    assert [result.model_id for result in store.list_for_session("sess-a")] == ["model-a", "model-b"]
    assert store.delete_session("sess-a") == 2
    assert store.list_for_session("sess-a") == []


def test_prediction_store_copies_frame_when_adding() -> None:
    store = PredictionStore()
    result = create_result()

    store.add(result)
    result.frame.loc[0, "prediction"] = "n1"

    stored = store.get_by_task(result.session_id, result.task)
    assert stored is not None
    assert stored.frame.loc[0, "prediction"] == "wake"


def test_prediction_store_add_many_is_atomic_on_conflict() -> None:
    store = PredictionStore()
    store.add(create_result(session_id="sess-a", model_id="model-a"))

    with pytest.raises(ValueError, match="already exist"):
        store.add_many(
            [
                create_result(session_id="sess-b", model_id="model-b"),
                create_result(session_id="sess-a", model_id="model-a"),
            ]
        )

    assert store.list_for_session("sess-b") == []


def test_prediction_store_rejects_duplicate_task_for_session() -> None:
    store = PredictionStore()
    store.add(create_result(model_id="model-a"))

    with pytest.raises(ValueError, match="task sleep_stage"):
        store.add(create_result(model_id="model-b"))


def test_mapped_records_are_cached_until_the_result_is_replaced() -> None:
    from dataclasses import replace
    from datetime import datetime

    from sleepwalker.server.results.predictions import PredictionKind

    store = PredictionStore()
    frame = pd.DataFrame({"time": [0.5, 1.5], "prob__no_arousal": [0.2, 0.9], "prob__arousal": [0.8, 0.1]})
    store.add(PredictionResult("sess-test", "model", "arousal", ("no_arousal", "arousal"), 1.0, frame, PredictionKind.EVENT))
    result = store.get_by_task("sess-test", "arousal")
    records, events = store.mapped(result, datetime(2025, 1, 1))

    assert store.mapped(result, datetime(2025, 1, 1))[0] is records
    assert [(event.start, event.end) for event in events] == [(0.0, 1.0)]
    assert store.mapped(replace(result), datetime(2025, 1, 1))[0] is not records
    store.delete_session("sess-test")
    assert store._mapped == {}
