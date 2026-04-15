"""Confusion-matrix-derived metric helpers."""

import numpy as np


def cohen_kappa_from_confusion_matrix(confusion_matrix):
    """Compute Cohen's kappa from a confusion matrix.

    Args:
        confusion_matrix: Square confusion matrix as a NumPy-like array.

    Returns:
        Cohen's kappa value. The current implementation returns `0.0` for
        several degenerate cases instead of propagating unstable divisions.
    """
    n_total = np.sum(confusion_matrix)
    p_o = np.trace(confusion_matrix) / n_total
    row_sums = np.sum(confusion_matrix, axis=1)
    col_sums = np.sum(confusion_matrix, axis=0)
    p_e = np.sum(row_sums * col_sums) / (n_total ** 2)

    if p_o < p_e or p_e > 1 or np.abs(1 - p_e) < 1e-3:
        return 0.0
    return (p_o - p_e) / (1 - p_e)


def f1_score_from_confusion_matrix(confusion_matrix, macro=False) -> float:
    """Compute micro- or macro-averaged F1 from a confusion matrix.

    Args:
        confusion_matrix: Square confusion matrix as a NumPy-like array.
        macro: Whether to average per-class F1 scores instead of computing a
            micro score.

    Returns:
        The requested F1 score as a float.
    """
    num_classes = confusion_matrix.shape[0]

    if macro:
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1_score = np.zeros(num_classes)

        for i in range(num_classes):
            true_positives = confusion_matrix[i, i]
            false_positives = np.sum(confusion_matrix[:, i]) - true_positives
            false_negatives = np.sum(confusion_matrix[i, :]) - true_positives

            precision[i] = 0.0 if true_positives == 0 and false_positives == 0 else true_positives / (true_positives + false_positives)
            recall[i] = 0.0 if true_positives == 0 and false_negatives == 0 else true_positives / (true_positives + false_negatives)
            f1_score[i] = 0.0 if precision[i] == 0.0 and recall[i] == 0.0 else 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

        return float(np.mean(f1_score))

    total_true_positives = np.sum(np.diagonal(confusion_matrix))
    total_false_positives = np.sum(confusion_matrix, axis=0) - np.diagonal(confusion_matrix)
    total_false_negatives = np.sum(confusion_matrix, axis=1) - np.diagonal(confusion_matrix)

    micro_precision = 0.0 if total_true_positives == 0 and np.sum(total_false_positives) == 0 else total_true_positives / (total_true_positives + np.sum(total_false_positives))
    micro_recall = 0.0 if total_true_positives == 0 and np.sum(total_false_negatives) == 0 else total_true_positives / (total_true_positives + np.sum(total_false_negatives))
    if micro_precision == 0.0 and micro_recall == 0.0:
        return 0.0
    return float(2 * (micro_precision * micro_recall) / (micro_precision + micro_recall))
