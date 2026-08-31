from pathlib import Path

import pytest
import torch

from iclr2026.scripts.evaluate_composition import GRAPH_METHODS, SYSTEMS, TASKS, build_zero_shot_package, common_cohort_aggregates, evaluation_config, read_config, replacement_name, validate_multitask_package, validate_single_task_package, write_outputs
from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment import PackagedModel
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel
from sleepwalker.models.ModelGraphClassifier import GraphNode, ModelGraphClassifier, load_graph_node, pair_datasets
from tools.train import read_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "iclr2026" / "configs" / "main"


class ConstantClassifier(BaseModel, ClassifierModel):
    def __init__(self, ts_len, classes):
        super().__init__()
        self.ts_len = ts_len
        self.classes = classes
        self.value = torch.nn.Parameter(torch.zeros(1, 1, len(classes)))

    def compute(self, inputs):
        return self.value.expand(inputs.shape[0], -1, -1)

    def input_spec(self):
        return (1, self.ts_len, 1), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": 1}


def test_composition_has_four_explicit_training_configs():
    paths = [CONFIG_DIR / f"{name}.yml" for name in GRAPH_METHODS]

    assert {path.stem for path in paths} == {"latent", "probability", "probability_osf", "probability_sleepfm", "sei"}
    for path in paths:
        config = read_yaml(path)
        assert config["model"]["name"] == "sleepwalker.models.ModelGraphClassifier.ModelGraphClassifier"
        assert config["model"]["method"] == GRAPH_METHODS[path.stem]
        assert config["data"]["name"] == "sleepwalker.models.ModelGraphClassifier.paired_dataset_from_packages"
        assert set(config["data"]["packages"]) == set(TASKS)
        assert config["model"]["nodes"]["name"] == "sleepwalker.models.ModelGraphClassifier.load_graph_nodes"
        assert config["model"]["nodes"]["packages"] == config["data"]["packages"]
        expected_package = f"iclr2026/models/{path.stem.replace('_', '-')}" if path.stem.startswith("probability_") else f"iclr2026/models/{path.stem}"
        assert config["run"]["package_path"] == expected_package


def test_evaluation_config_defines_the_fixed_complete_night_protocol():
    config = {
        "data": {"label": "HSP", "split": {"file": "split.yml"}, "strict": False, "dataset": {}},
        "systems": {},
        "zero_shot": [],
        "test": {"batch_size": 4, "output": "old"},
    }

    result = evaluation_config(config, Path("result.jsonl"), device="cpu")

    assert result["data"]["strict"] is False
    assert result["data"]["dataset"]["group_sampling_strategy"] == "first"
    assert "systems" not in result
    assert "zero_shot" not in result
    assert [analysis["name"] for analysis in result["analyses"]] == ["complete_night"]
    assert result["analyses"][0]["pipeline"][1]["labels"] == ["wake", "n1", "n2", "n3", "rem"]
    assert result["test"]["output"] == "result.jsonl"


def test_package_validation_uses_loaded_package_apis():
    dataset = UnlabelledDataset(channels=[ChannelConfig("signal", ["signal"])], sample_frequency=10, total_input="1s", target_resolution="1s", stride="1s")
    single = PackagedModel(name="sleep", task="sleep", model=ConstantClassifier(10, ["a", "b"]), dataset=dataset, classification_contract={"type": "single-head-multiclass", "classes": ["a", "b"], "sequence_len": 1})
    validate_single_task_package(single, "sleep")
    with pytest.raises(ValueError, match="arousal"):
        validate_single_task_package(single, "arousal")

    outputs = {task: {"classes": [f"{task}-negative", f"{task}-positive"], "sequence_len": 1} for task in TASKS}
    node = GraphNode(model=ConstantClassifier(10, ["a", "b"]), outputs=outputs)
    graph = ModelGraphClassifier(nodes={"shared": node}, method="probability")
    contract = {"type": "multitask", "tasks": {task: {"classes": output["classes"], "n_steps": 1} for task, output in outputs.items()}}
    package = PackagedModel(name="graph", task="multitask", model=graph, dataset=dataset, classification_contract=contract)
    validate_multitask_package(package, "probability")
    with pytest.raises(ValueError, match="'sei'"):
        validate_multitask_package(package, "sei")


def test_checked_in_evaluation_config_is_small_and_package_driven():
    config = read_config(CONFIG_DIR / "test.yml")

    assert set(config["systems"]) == set(SYSTEMS)
    assert "analyses" not in config
    assert "channels" not in config["data"]["dataset"]
    assert all(set(replacement) == {"task", "package"} for replacement in config["zero_shot"])
    assert all(str(path).startswith("iclr2026/models/") for system in ("sleepwalker", "osf", "sleepfm") for path in config["systems"][system].values())
    assert all(str(config["systems"][name]).startswith("iclr2026/models/") for name in (*GRAPH_METHODS, "multitask"))
    assert all(replacement["package"].startswith("iclr2026/models/") for replacement in config["zero_shot"])


def test_zero_shot_system_is_an_ordinary_paired_model_graph():
    sleep_classes = ["wake", "sleep"]
    arousal_classes = ["no", "yes"]
    sleep_dataset = UnlabelledDataset(channels=[ChannelConfig("sleep", ["EEG"])], sample_frequency=10, total_input="60s", target_resolution="60s", stride="30s")
    arousal_dataset = UnlabelledDataset(channels=[ChannelConfig("arousal", ["EMG"])], sample_frequency=10, total_input="20s", target_resolution="20s", stride="20s")
    sleep_expert = PackagedModel(name="sleepwalker-sleep", task="sleep", model=ConstantClassifier(600, sleep_classes), dataset=sleep_dataset, classification_contract={"type": "single-head-multiclass", "classes": sleep_classes, "sequence_len": 1})
    arousal_expert = PackagedModel(name="sleepwalker-arousal", task="arousal", model=ConstantClassifier(200, arousal_classes), dataset=arousal_dataset, classification_contract={"type": "single-head-multiclass", "classes": arousal_classes, "sequence_len": 1})
    nodes = {"sleep": load_graph_node(sleep_expert), "arousal": load_graph_node(arousal_expert)}
    graph = ModelGraphClassifier(nodes=nodes, method="probability", edges=[["sleep", "arousal"]])
    shared_dataset = pair_datasets({"sleep": sleep_dataset, "arousal": arousal_dataset})
    contract = {"type": "multitask", "tasks": {"sleep": {"classes": sleep_classes, "n_steps": 1, "target_resolution": "60s", "target_offset": "0s"}, "arousal": {"classes": arousal_classes, "n_steps": 1, "target_resolution": "20s", "target_offset": "20s"}}}
    base = PackagedModel(name="probability", task="multitask", model=graph, dataset=shared_dataset, classification_contract=contract)
    replacement_dataset = UnlabelledDataset(channels=[ChannelConfig("arousal", ["EEG"])], sample_frequency=5, total_input="20s", target_resolution="20s", stride="20s")
    replacement = PackagedModel(name="osf", task="arousal", model=ConstantClassifier(100, arousal_classes), dataset=replacement_dataset, classification_contract={"type": "single-head-multiclass", "classes": arousal_classes, "sequence_len": 1})

    result = build_zero_shot_package(base, {"sleep": sleep_expert, "arousal": arousal_expert}, replacement, "arousal", "zero-shot")

    assert isinstance(result.model, ModelGraphClassifier)
    assert result.model.input_spec()[1]["layout"] == "mapping"
    assert result.dataset.datasets["arousal"].sample_frequency == 5
    assert result.dataset.datasets["sleep"].sample_frequency == 10
    assert torch.equal(result.model.heads["arousal"].weight, graph.heads["arousal"].weight)


def aggregate(task, matrices):
    return {
        "record_type": "aggregate",
        "dataset": "HSP",
        "analysis": "complete_night",
        "task": task,
        "classes": ["negative", "positive"],
        "n_patients_requested": 4,
        "n_patients_initialized": 3,
        "n_patients_evaluated": len(matrices),
        "patient_confusion_matrices": matrices,
    }


def test_comparison_recomputes_metrics_on_exact_patient_intersection():
    records = {
        "first": [aggregate(task, {"a": [[1, 0], [0, 1]], "b": [[2, 0], [0, 1]]}) for task in TASKS],
        "second": [aggregate(task, {"b": [[1, 1], [0, 2]], "c": [[1, 0], [1, 1]]}) for task in TASKS],
    }

    aggregates, cohort_rows = common_cohort_aggregates(records)

    assert all(record["n_patients_common"] == 1 for record in aggregates)
    assert all(record["patient_confusion_matrices"].keys() == {"b"} for record in aggregates)
    assert all(row["patients_common"] == 1 for row in cohort_rows)
    first_sleep = next(record for record in aggregates if record["system"] == "first" and record["task"] == "sleep")
    assert first_sleep["metrics"]["confusion_matrix"] == [[2, 0], [0, 1]]


def test_main_output_uses_macro_f1_for_every_task(tmp_path):
    aggregates = [{"system": "independent", "analysis": "complete_night", "task": task, "metrics": {"f1_macro": 0.5 + index / 10}} for index, task in enumerate(TASKS)]
    cohort_rows = [{"system": "independent", "dataset": "HSP", "task": "sleep", "patients_requested": 1, "patients_initialized": 1, "patients_system_evaluated": 1, "patients_common": 1}]

    write_outputs(aggregates, cohort_rows, ["independent"], tmp_path, overwrite=False)

    csv_text = (tmp_path / "main_results.csv").read_text()
    latex = (tmp_path / "main_results.tex").read_text()
    assert "system,sleep,arousal,breathing,desaturation" in csv_text
    assert "independent,0.5,0.6,0.7,0.8" in csv_text
    assert "independent & 0.500 & 0.600 & 0.700 & 0.800" in latex
    assert replacement_name("sleep", "osf") == "zero_shot_sleep_osf"
    assert replacement_name("sleep", "sleepfm") == "zero_shot_sleep_sleepfm"
