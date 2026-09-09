from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from iclr2026.scripts.plots_and_tables import aggregate_systems, common_cohort, patient_is_complete, write_outputs
from iclr2026.replacement import build_zero_shot_package
from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment import PackagedModel
from sleepwalker.deployment.evaluation import condition_endpoint, validate_dependencies
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel
from sleepwalker.models.ModelGraphClassifier import ModelGraphClassifier, PairedDataset, load_graph_node
from tools.evaluate_system import analyze_patient, read_config
from tools.train import read_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "iclr2026" / "configs" / "main" / "train"
EVAL_DIR = REPO_ROOT / "iclr2026" / "configs" / "main" / "eval"
TASKS = ("sleep", "arousal", "breathing", "desaturation")


class ConstantClassifier(BaseModel, ClassifierModel):
    def __init__(self, ts_len, classes, sequence_len=1):
        super().__init__()
        self.ts_len = ts_len
        self.classes = classes
        self.value = torch.nn.Parameter(torch.zeros(1, sequence_len, len(classes)))

    def compute(self, inputs):
        return self.value.expand(inputs.shape[0], -1, -1)

    def input_spec(self):
        return (1, self.ts_len, 1), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": 1}


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


def test_training_configs_moved_and_evaluations_cover_every_system():
    assert not (TRAIN_DIR.parent / "test.yml").exists()
    train_paths = sorted(TRAIN_DIR.glob("*.yml"))
    assert len(train_paths) == 11
    for path in train_paths:
        config = read_yaml(path)
        assert config["model"]["name"] == "sleepwalker.models.ModelGraphClassifier.ModelGraphClassifier"
        packages = config["model"]["nodes"]["packages"]
        assert set(packages) == set(TASKS)
        assert "packages" not in config["data"]
        assert config["trainer"]["eval_every"] == 0
        assert config["trainer"]["return_best"] is False
        assert config["run"]["test_repeats"] == []

    configs = [read_config(path) for path in sorted(EVAL_DIR.glob("*.yml"))]
    assert len(configs) == 26
    assert len({config["system"]["name"] for config in configs}) == 26
    assert {config["system"]["type"] for config in configs} == {"independent", "package", "factory"}
    assert all(len(config["dependencies"]) == 3 for config in configs)
    assert all(config["data"]["dataset"]["group_sampling_strategy"] == "first" for config in configs)


def test_zero_shot_system_is_an_ordinary_paired_graph():
    sleep_classes = ["wake", "n1"]
    event_classes = ["no_arousal", "arousal"]
    sleep_dataset = UnlabelledDataset(channels=[ChannelConfig("sleep", ["EEG"])], sample_frequency=10, total_input="60s", stride="30s")
    event_dataset = UnlabelledDataset(channels=[ChannelConfig("arousal", ["EMG"])], sample_frequency=10, total_input="20s", stride="20s")
    experts = {
        "sleep": PackagedModel(name="sleep", task="sleep", model=ConstantClassifier(600, sleep_classes), dataset=sleep_dataset, classification_contract={"type": "single-head-multiclass", "classes": sleep_classes, "sequence_len": 1, "target_resolution": "60s"}),
        "arousal": PackagedModel(name="arousal", task="arousal", model=ConstantClassifier(200, event_classes), dataset=event_dataset, classification_contract={"type": "single-head-multiclass", "classes": event_classes, "sequence_len": 1, "target_resolution": "20s"}),
    }
    for task in ("breathing", "desaturation"):
        classes = ["event", "none"]
        experts[task] = PackagedModel(name=task, task=task, model=ConstantClassifier(200, classes), dataset=event_dataset, classification_contract={"type": "single-head-multiclass", "classes": classes, "sequence_len": 1, "target_resolution": "20s"})
    task_contracts = {
        task: {"classes": package.classification_contract["classes"], "n_steps": 1, "target_resolution": "20s", "target_offset": "20s"}
        for task, package in experts.items()
    }
    nodes = {task: load_graph_node(package, graph_outputs={task: task_contracts[task]}) for task, package in experts.items()}
    graph = ModelGraphClassifier(nodes=nodes, method="probability", edges=[["sleep", "arousal"]])
    reference = UnlabelledDataset(channels=[ChannelConfig("sleep", ["EEG"])], sample_frequency=10, total_input="60s", stride="30s")
    shared_dataset = PairedDataset({task: package.dataset for task, package in experts.items()}, base=reference, input_offsets={task: node.input_offsets for task, node in nodes.items()})
    base = PackagedModel(name="probability", task="multitask", model=graph, dataset=shared_dataset, classification_contract={"type": "multitask", "tasks": task_contracts})
    replacement = PackagedModel(name="replacement", task="arousal", model=ConstantClassifier(100, event_classes, sequence_len=2), dataset=UnlabelledDataset(channels=[ChannelConfig("arousal", ["EEG"])], sample_frequency=5, total_input="20s", stride="20s"), classification_contract={"type": "single-head-multiclass", "classes": event_classes, "sequence_len": 2, "target_resolution": "20s"})

    result = build_zero_shot_package(base_package=base, experts=experts, replacement={"task": "arousal", "package": replacement}, name="zero-shot")

    assert isinstance(result.model, ModelGraphClassifier)
    assert result.model.input_spec()[1]["layout"] == "mapping"
    assert result.dataset.datasets["arousal"].sample_frequency == 5
    assert torch.equal(result.model.heads["arousal"].weight, graph.heads["arousal"].weight)


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


def patient_record(system, matrices, missing_fraction=0.0):
    return {
        "record_type": "patient",
        "system": system,
        "patient": "patient",
        "coverage": {"model": {"initialized": True, "missing_fraction": missing_fraction}},
        "tasks": {
            task: {"classes": ["negative", "positive"], "metrics": {"confusion_matrix": matrix}}
            for task, matrix in zip(TASKS, matrices)
        },
    }


def test_common_cohort_uses_global_one_percent_threshold_and_aggregates():
    matrix = [[1, 0], [0, 1]]
    evaluations = {
        "first": {
            "a": {**patient_record("first", [matrix] * 4), "patient": "a"},
            "b": {**patient_record("first", [matrix] * 4, missing_fraction=0.02), "patient": "b"},
        },
        "second": {
            "a": {**patient_record("second", [matrix] * 4), "patient": "a"},
            "b": {**patient_record("second", [matrix] * 4), "patient": "b"},
        },
    }

    common, qualified = common_cohort(["first", "second"], evaluations, 0.01)
    aggregates = aggregate_systems(["first", "second"], evaluations, common)

    assert common == {"a"}
    assert qualified["first"] == {"a"}
    assert len(aggregates) == 8
    assert all(record["n_patients"] == 1 for record in aggregates)


def test_patient_completeness_rejects_missing_tasks():
    record = patient_record("system", [[[1, 0], [0, 1]]] * 4)
    assert patient_is_complete(record, 0.01)
    del record["tasks"]["sleep"]
    assert not patient_is_complete(record, 0.01)


def test_reporting_writes_main_and_independent_tables(tmp_path):
    matrix = [[1, 0], [0, 1]]
    systems = ["independent_sleepwalker", "probability"]
    evaluations = {
        system: {"patient": patient_record(system, [matrix] * 4)}
        for system in systems
    }
    aggregates = aggregate_systems(systems, evaluations, {"patient"})

    write_outputs(tmp_path, systems, evaluations, {"patient"}, {system: {"patient"} for system in systems}, aggregates, 0.01, False)

    assert "Independent Sleepwalker experts" in (tmp_path / "independent_results.tex").read_text(encoding="utf-8")
    assert "Probability graph (Sleepwalker)" in (tmp_path / "main_results.tex").read_text(encoding="utf-8")
    assert (tmp_path / "main_results.pdf").stat().st_size > 0
