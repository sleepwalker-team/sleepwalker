from __future__ import annotations

import pandas as pd
import pytest

from sleepwalker.deployment.evaluation import condition_endpoint, read_prediction_feather, validate_dependencies, write_prediction_feather
from tools.evaluate_system import analyze_patient


TASKS = ("sleep", "arousal", "breathing", "desaturation")


def prediction_frame(times, classes, predictions, targets=None, annotations=None):
    rows = []
    for index, (time, prediction) in enumerate(zip(times, predictions)):
        row = {"time": pd.Timestamp(time), "target": 0 if targets is None else targets[index], "valid": True}
        for class_name in classes:
            row[f"prob__{class_name}"] = float(class_name == prediction)
        for label, values in (annotations or {}).items():
            row[f"annotation__{label}"] = values[index]
        rows.append(row)
    return pd.DataFrame(rows)


def test_prediction_feather_roundtrip_can_select_one_patient(tmp_path):
    first = prediction_frame(["2020-01-01 00:00:00"], ["no", "yes"], ["no"])
    first.insert(0, "task", "arousal")
    first.insert(0, "patient", "first.edf")
    second = prediction_frame(["2020-01-01 00:00:01"], ["no", "yes"], ["yes"])
    second.insert(0, "task", "arousal")
    second.insert(0, "patient", "second.edf")
    path = tmp_path / "predictions.feather"

    write_prediction_feather(path, pd.concat([first, second], ignore_index=True))
    restored = read_prediction_feather(path, patients=["second.edf"])

    assert restored["patient"].unique().tolist() == ["second.edf"]
    assert restored["task"].tolist() == second["task"].tolist()
    assert restored["time"].tolist() == second["time"].tolist()
    assert restored["target"].tolist() == second["target"].tolist()
    assert restored[["prob__no", "prob__yes"]].to_numpy().tolist() == second[["prob__no", "prob__yes"]].to_numpy().tolist()


def test_condition_endpoint_uses_fifty_percent_interval_coverage():
    parent = prediction_frame(
        ["2020-01-01 00:00:00", "2020-01-01 00:00:30"],
        ["wake", "n1"],
        ["wake", "n1"],
    )
    child = prediction_frame(
        ["2020-01-01 00:00:20", "2020-01-01 00:00:25", "2020-01-01 00:00:30"],
        ["regular breathing", "apnea"],
        ["apnea", "apnea", "apnea"],
    )

    conditioned = condition_endpoint(child, parent, child_resolution=pd.Timedelta("10s"), parent_resolution=pd.Timedelta("30s"), active_classes=["n1"], target_default="regular breathing", minimum_coverage=0.5)

    assert conditioned[["prob__regular breathing", "prob__apnea"]].to_numpy().argmax(axis=1).tolist() == [0, 1, 1]


def test_dependency_graph_rejects_cycles_and_unknown_classes():
    classes = {"first": ["off", "on"], "second": ["off", "on"]}
    with pytest.raises(ValueError, match="acyclic"):
        validate_dependencies([
            {"source": "first", "target": "second", "active_classes": ["on"], "target_default": "off"},
            {"source": "second", "target": "first", "active_classes": ["on"], "target_default": "off"},
        ], classes)
    with pytest.raises(ValueError, match="invalid active classes"):
        validate_dependencies([
            {"source": "first", "target": "second", "active_classes": ["missing"], "target_default": "off"},
        ], classes)


def test_patient_analysis_propagates_sleep_and_breathing_errors():
    annotations = {label: [1.0, 1.0] if label == "n1" else [0.0, 0.0] for label in ("wake", "n1", "n2", "n3", "rem")}
    start = pd.Timestamp("2020-01-01")
    frames = []
    task_values = {
        "sleep": (["wake", "n1", "n2", "n3", "rem"], ["wake", "n1"], [1, 1]),
        "arousal": (["no_arousal", "arousal"], ["arousal", "arousal"], [1, 1]),
        "breathing": (["apnea", "hypopnea", "regular breathing"], ["apnea", "regular breathing"], [0, 0]),
        "desaturation": (["desaturation", "no desaturation"], ["desaturation", "desaturation"], [0, 0]),
    }
    for task, (classes, predictions, targets) in task_values.items():
        frame = prediction_frame([start, start + pd.Timedelta("30s")], classes, predictions, targets=targets, annotations=annotations)
        frame.insert(0, "task", task)
        frame.insert(0, "patient", "patient")
        frames.append(frame)
    resolutions = {task: pd.Timedelta("30s") for task in TASKS}
    classes_by_task = {task: values[0] for task, values in task_values.items()}
    dependencies = validate_dependencies([
        {"source": "sleep", "target": "breathing", "active_classes": ["n1", "n2", "n3", "rem"], "target_default": "regular breathing", "minimum_coverage": 0.5},
        {"source": "sleep", "target": "arousal", "active_classes": ["n1", "n2", "n3", "rem"], "target_default": "no_arousal", "minimum_coverage": 0.5},
        {"source": "breathing", "target": "desaturation", "active_classes": ["apnea", "hypopnea"], "target_default": "no desaturation", "minimum_coverage": 0.5},
    ], classes_by_task)

    result = analyze_patient(pd.concat(frames, ignore_index=True), resolutions, classes_by_task, [], dependencies)

    assert result["arousal"]["metrics"]["confusion_matrix"] == [[0, 0], [1, 1]]
    assert result["breathing"]["metrics"]["confusion_matrix"][0] == [0, 0, 2]
    assert result["desaturation"]["metrics"]["confusion_matrix"] == [[0, 2], [0, 0]]
