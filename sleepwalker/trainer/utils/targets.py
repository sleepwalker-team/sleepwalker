"""Helpers for turning label windows into multiclass training targets.

These functions are used by dataset callbacks and are covered directly by
``tests/test_targets.py``. They currently implement a conservative target
construction path: ambiguous windows are rejected rather than force-assigned.
"""

from typing import Sequence

import pandas as pd
import torch


def resolve_multiclass_index(target, default_idx, min_event_seconds, raise_error=True):
    """Resolve one active class index from per-class event durations.

    Args:
        target: One-dimensional tensor-like collection containing one duration
            per class.
        default_idx: Class index to return when no class exceeds the minimum
            duration threshold.
        min_event_seconds: Minimum duration required for a class to count as
            active.
        raise_error: Whether ambiguous targets should raise ``ValueError``.

    Returns:
        The resolved class index or ``None`` when ambiguity is tolerated.

    Raises:
        ValueError: If multiple classes are active or no class is active and no
            default index is provided while ``raise_error`` is true.
    """
    active = target > min_event_seconds
    active_sum = int(active.sum().item())

    if raise_error and active_sum > 1:
        raise ValueError("Multiple active classes found.")
    if active_sum == 1:
        return int(active.nonzero(as_tuple=False).item())
    if active_sum == 0:
        if default_idx is not None:
            return int(default_idx)
        if raise_error:
            raise ValueError("Ambiguous class labels found with no active class and no default_idx.")
        return None
    return None

def _passes_filters(target, filters) -> bool:
    """Return whether a target window satisfies all configured filters.

    `filters` is a list of dicts with:
    - `columns`: label columns inspected together
    - `percentage`: threshold in `[0, 1]`
    - `mode`:
      - `"min"` keeps the sample only if the fraction of timesteps where any
        of `columns` is active is at least `percentage`
      - `"max"` keeps the sample only if that fraction is at most
        `percentage`

    Examples
    --------
    Require at least 50% sleep:
    `{"columns": ["n1", "n2", "n3", "rem"], "percentage": 0.5, "mode": "min"}`

    Reject windows with any artifact or movement:
    `{"columns": ["artifact", "movement"], "percentage": 0.0, "mode": "max"}`
    """
    if filters is None:
        return True

    for spec in filters:
        value = float(target.reindex(columns=list(spec["columns"]), fill_value=0).any(axis=1).mean())
        threshold = float(spec.get("percentage", 0.0))
        if threshold < 0.0:
            raise ValueError("Threshold cannot be negative.")
        mode = spec.get("mode", "max")

        if mode == "min":
            if value < threshold:
                return False
        elif mode == "max":
            if value > threshold:
                return False
        else:
            raise ValueError(f"Unknown filter mode '{mode}'.")

    return True


def _build_multiclass_onehot(target, target_classes: Sequence[str], percentage: float):
    """Convert a label window into a one-hot vector in `target_classes` order.

    A class is active when it covers at least `percentage` of the window.
    If no present class is active, the function accepts a single implicit
    fallback class: exactly one class from `target_classes` may be missing from
    `target.columns`, and that class becomes the default label. This is useful
    for tasks such as `["desaturation", "no desaturation"]`.
    """
    target_classes = list(target_classes)
    if len(target_classes) == 0:
        raise ValueError("target_classes must not be empty.")
    
    present_classes = [label for label in target_classes if label in target.columns]
    fallback_classes = [label for label in target_classes if label not in target.columns]

    freq = pd.to_timedelta(target.index.freq).total_seconds()
    threshold = len(target) * freq * percentage
    durations = torch.tensor(
        target.reindex(columns=present_classes, fill_value=0).sum().to_numpy() * freq,
        dtype=torch.float32,
    )
    active = durations >= threshold
    active_sum = int(active.sum().item())

    if active_sum > 1:
        #raise ValueError("Multiple active classes found.")
        return None
    if active_sum == 1:
        target_label = present_classes[int(active.nonzero(as_tuple=False).item())]
    elif len(fallback_classes) == 1:
        target_label = fallback_classes[0]
    else:
        # raise ValueError("Ambiguous class labels found with no active class and no unique fallback class.")
        return None

    onehot = torch.zeros(len(target_classes), dtype=torch.float32)
    onehot[target_classes.index(target_label)] = 1.0
    return onehot

def prepare_multiclass_target(
    target,
    target_extra=None,
    patient=None,
    time=None,
    *,
    target_classes: Sequence[str],
    percentage: float = 0.5,
    filters=None,
):
    """Build one-hot multiclass targets from a label-activity window.

    Args:
        target: Primary label window as a time-indexed DataFrame with one
            column per label and binary activity values.
        target_extra: Optional second label source for the same window. It is
            transformed independently and does not reject an otherwise valid
            primary target.
        patient: Unused callback argument kept for dataset API compatibility.
        time: Unused callback argument kept for dataset API compatibility.
        target_classes: Output class order for the returned one-hot vectors.
        percentage: Minimum fraction of the window a class must cover to be
            considered active.
        filters: Optional filter specifications applied before target
            construction.

    Returns:
        A dictionary containing ``target`` and optionally ``target_extra``, or
        ``None`` when the window is filtered out or cannot be resolved
        unambiguously.

    Notes:
        Tests confirm the current fallback behavior: if exactly one target class
        is absent from the input columns, that class may serve as the implicit
        negative/default class.
    """
    if target is None:
        return None
    if not _passes_filters(target, filters):
        return None

    try:
        target_onehot = _build_multiclass_onehot(target, target_classes=target_classes, percentage=percentage)
    except ValueError:
        return None

    if target_onehot is None:
        return None

    item = {"target": target_onehot}
    if target_extra is None:
        return item

    try:
        target_extra_onehot = _build_multiclass_onehot(
            target_extra,
            target_classes=target_classes,
            percentage=percentage,
        )
    except ValueError:
        target_extra_onehot = None

    if target_extra_onehot is not None:
        item["target_extra"] = target_extra_onehot
    return item
