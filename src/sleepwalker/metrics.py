"""Reusable metric calculations derived from confusion matrices."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def probability_metrics(targets, probabilities, classes):
    """One-vs-rest AUROC/AP in declared class order, using sklearn definitions.

    No positives makes both metrics undefined; only positives gives AP=1 and
    undefined AUROC. Macros include every declared class, returning None if a
    constituent is undefined. JSON output therefore preserves missing curves.
    """
    classes = list(classes)
    targets, probabilities = np.asarray(targets), np.asarray(probabilities)
    if len(classes) < 2 or len(set(classes)) != len(classes) or any(not isinstance(label, str) or not label for label in classes):
        raise ValueError("classes must contain distinct nonempty labels.")
    if targets.ndim != 1 or probabilities.shape != (len(targets), len(classes)):
        raise ValueError("Expected targets [N] and probabilities [N,C].")
    if targets.dtype.kind not in "biuf" or not np.isfinite(targets).all() or np.any(targets != np.floor(targets)) or np.any(targets < 0) or np.any(targets >= len(classes)):
        raise ValueError("Targets must be integer class IDs in the declared vocabulary.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probabilities must be finite, in [0,1], and sum to one per row.")
    per_class = {}
    for index, label in enumerate(classes):
        positive = targets == index
        support, negative = int(positive.sum()), int((~positive).sum())
        per_class[label] = {
            "support": support, "negative_support": negative,
            "auroc": float(roc_auc_score(positive, probabilities[:, index])) if support and negative else None,
            "average_precision": float(average_precision_score(positive, probabilities[:, index])) if support else None,
            "auroc_reason": None if support and negative else "no_positive_targets" if not support else "no_negative_targets",
            "average_precision_reason": None if support else "no_positive_targets",
        }
    result = {"per_class": per_class, "macro_denominator": len(classes), "macro_policy": "all_declared_classes"}
    for field in ["auroc", "average_precision"]:
        values = [value[field] for value in per_class.values()]
        defined = sum(value is not None for value in values)
        result[f"{field}_macro"] = float(np.mean(values)) if defined == len(values) else None
        result[f"{field}_defined_classes"] = defined
        result[f"{field}_macro_reason"] = None if defined == len(values) else "one_or_more_classes_undefined"
    return result


def validate_confusion_matrix(confusion_matrix) -> np.ndarray:
    matrix = np.asarray(confusion_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square confusion matrix, got shape {matrix.shape}.")
    if np.any(matrix < 0):
        raise ValueError("Confusion-matrix entries must be non-negative.")
    return matrix


def accuracy_from_confusion_matrix(confusion_matrix) -> float:
    matrix = validate_confusion_matrix(confusion_matrix)
    total = matrix.sum()
    return 0.0 if total == 0 else float(np.trace(matrix) / total)


def precision_from_confusion_matrix(confusion_matrix) -> np.ndarray:
    matrix = validate_confusion_matrix(confusion_matrix)
    true_positive = np.diag(matrix).astype(float)
    denominator = matrix.sum(axis=0)
    return np.divide(true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)


def recall_from_confusion_matrix(confusion_matrix) -> np.ndarray:
    matrix = validate_confusion_matrix(confusion_matrix)
    true_positive = np.diag(matrix).astype(float)
    denominator = matrix.sum(axis=1)
    return np.divide(true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)


def f1_per_class_from_confusion_matrix(confusion_matrix) -> np.ndarray:
    precision = precision_from_confusion_matrix(confusion_matrix)
    recall = recall_from_confusion_matrix(confusion_matrix)
    return np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=precision + recall > 0)


def support_from_confusion_matrix(confusion_matrix) -> np.ndarray:
    return validate_confusion_matrix(confusion_matrix).sum(axis=1)


def cohen_kappa_from_confusion_matrix(confusion_matrix) -> float:
    matrix = validate_confusion_matrix(confusion_matrix)
    total = matrix.sum()
    if total == 0:
        return 0.0
    observed = np.trace(matrix) / total
    expected = np.dot(matrix.sum(axis=1), matrix.sum(axis=0)) / (total * total)
    if np.isclose(expected, 1.0):
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def f1_score_from_confusion_matrix(confusion_matrix, macro=False) -> float:
    matrix = validate_confusion_matrix(confusion_matrix)
    if macro:
        return float(f1_per_class_from_confusion_matrix(matrix).mean())
    return accuracy_from_confusion_matrix(matrix)
