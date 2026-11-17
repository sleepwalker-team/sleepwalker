from io import StringIO
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from sleepwalker.deployment import evaluation
from sleepwalker.metrics import probability_metrics


@pytest.mark.parametrize("n_classes", [2, 3])
def test_probability_metrics_match_sklearn_with_ties(n_classes):
    rng = np.random.default_rng(17)
    scores = rng.integers(1, 5, size=(60, n_classes)).astype(float)
    scores /= scores.sum(axis=1, keepdims=True)
    targets = np.arange(60) % n_classes
    classes = [f"class-{index}" for index in range(n_classes)]
    metrics = probability_metrics(targets, scores, classes)
    for index, label in enumerate(classes):
        assert metrics["per_class"][label]["auroc"] == pytest.approx(roc_auc_score(targets == index, scores[:, index]))
        assert metrics["per_class"][label]["average_precision"] == pytest.approx(average_precision_score(targets == index, scores[:, index]))


def test_absent_class_and_single_class_policies_have_fixed_macro_denominator():
    metrics = probability_metrics([1, 1], [[0.2, 0.8], [0.4, 0.6]], ["negative", "positive"])
    assert metrics["per_class"]["positive"]["average_precision"] == 1
    assert metrics["per_class"]["negative"]["average_precision"] is None
    assert metrics["per_class"]["positive"]["auroc_reason"] == "no_negative_targets"
    assert metrics["per_class"]["negative"]["auroc_reason"] == "no_positive_targets"
    assert metrics["macro_denominator"] == 2
    assert metrics["auroc_macro"] is metrics["average_precision_macro"] is None
    assert metrics["average_precision_defined_classes"] == 1
    assert "null" in json.dumps(metrics, allow_nan=False)
    empty = probability_metrics([], np.empty((0, 2)), ["negative", "positive"])
    assert empty["auroc_defined_classes"] == empty["average_precision_defined_classes"] == 0


@pytest.mark.parametrize("targets,scores,classes", [([2], [[0.5, 0.5]], ["a", "b"]), ([0.5], [[0.5, 0.5]], ["a", "b"]), ([0], [[np.nan, 0.5]], ["a", "b"]), ([0], [[0.2, 0.2]], ["a", "b"]), ([0], [[1.1, -0.1]], ["a", "b"]), ([0], [[0.5, 0.5]], ["a", "a"])])
def test_invalid_probability_inputs_fail(targets, scores, classes):
    with pytest.raises(ValueError):
        probability_metrics(targets, scores, classes)


def test_pooled_curves_differ_from_patient_means(monkeypatch):
    classes = ["negative", "positive"]
    package = SimpleNamespace(name="synthetic", task="event", classification_contract={"type": "single-head-multiclass", "classes": classes})
    dataset = SimpleNamespace(edf_files=[SimpleNamespace(path=patient) for patient in ["a.edf", "b.edf"]], get_n_patients=lambda: 2)
    monkeypatch.setattr(evaluation, "read_edf_meta", lambda path: {"signals": ["signal"]})
    frames = []
    for patient, probabilities in [("a.edf", [0.1, 0.2]), ("b.edf", [0.8, 0.9])]:
        frames.append(pd.DataFrame({"patient": patient, "task": "event", "valid": True, "time": pd.date_range("2025-01-01", periods=2, freq="30s"), "target": [0, 1], "prob__negative": 1 - np.asarray(probabilities), "prob__positive": probabilities}))
    output = StringIO()
    evaluation.evaluate_predictions(output, package, dataset, pd.concat(frames), [("raw", [])], dataset_label="synthetic", fold=None, n_patients_requested=2)
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    aggregate = records[-1]
    assert aggregate["metrics"]["per_class"]["positive"]["auroc"] == 0.75
    assert aggregate["patient_mean_metrics"]["per_class"]["positive"]["auroc"] == 1
    asymmetric = probability_metrics([0, 0, 0, 1], [[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.4, 0.6]], classes)
    assert asymmetric["per_class"]["positive"]["average_precision"] != asymmetric["average_precision_macro"]
