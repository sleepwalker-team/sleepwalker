"""Reusable metric calculations derived from confusion matrices."""

import numpy as np


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
