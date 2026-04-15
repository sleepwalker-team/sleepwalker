"""Formatting helpers for confusion-matrix logging."""

import numpy as np
import pandas as pd

from .metrics import cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix


def format_confusion_table(labels, confusion_matrix, title: str | None = None) -> list[str]:
    """Render one confusion matrix and summary metrics as aligned text lines.

    Args:
        labels: Class labels shown on rows and columns.
        confusion_matrix: Square confusion matrix.
        title: Optional block title.

    Returns:
        A list of equal-width strings suitable for side-by-side rendering.
    """
    short_labels = [str(label)[:5] for label in labels]
    df = pd.DataFrame(confusion_matrix, index=short_labels, columns=short_labels)
    table_lines = df.to_string().splitlines()

    total = confusion_matrix.sum()
    if total > 0:
        accuracy = confusion_matrix.trace() / total * 100.0
        f1_micro = f1_score_from_confusion_matrix(confusion_matrix, macro=False)
        f1_macro = f1_score_from_confusion_matrix(confusion_matrix, macro=True)
        kappa = cohen_kappa_from_confusion_matrix(confusion_matrix)
    else:
        accuracy = 0.0
        f1_micro = 0.0
        f1_macro = 0.0
        kappa = 0.0

    lines = [
        *([title] if title is not None else []),
        *table_lines,
        f"acc: {accuracy:5.2f}% f1(mi/ma): {f1_micro:1.4f} / {f1_macro:1.4f} kappa: {kappa:1.4f}",
    ]
    width = max(len(line) for line in lines)
    return [line.ljust(width) for line in lines]


def render_confusion_table_grid(blocks, header: str, n_cols: int = 3) -> str:
    """Lay out several formatted confusion tables in a text grid."""
    if len(blocks) == 0:
        return ""

    separator = "   "
    output_lines = [header]
    for start in range(0, len(blocks), n_cols):
        row_blocks = blocks[start:start + n_cols]
        height = max(len(block) for block in row_blocks)
        padded_blocks = [block + [" " * len(block[0])] * (height - len(block)) for block in row_blocks]
        output_lines.extend([
            separator.join(block[line_idx] for block in padded_blocks).rstrip()
            for line_idx in range(height)
        ])
        if start + n_cols < len(blocks):
            output_lines.append("")
    return "\n".join(output_lines)
