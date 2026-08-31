"""Helpers for turning label windows into multiclass training targets.

These functions are used by dataset callbacks and are covered directly by
``tests/test_targets.py``. They currently implement a conservative target
construction path: ambiguous windows are rejected rather than force-assigned.
"""

from collections import OrderedDict
import random
from typing import Optional, Sequence

import numpy as np
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

def passes_filters(target, filters) -> bool:
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


def build_multiclass_onehot(target, target_classes: Sequence[str], percentage: float):
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


# Reference implementation retained temporarily for comparison.
# def build_soft_multiclass_target(target, target_classes: Sequence[str]):
#     """Return class probabilities from the temporal coverage in one step."""
#     target_classes = list(target_classes)
#     if len(target_classes) == 0:
#         raise ValueError("target_classes must not be empty.")
#
#     present_classes = [label for label in target_classes if label in target.columns]
#     fallback_classes = [label for label in target_classes if label not in target.columns]
#     activity = target.reindex(columns=present_classes, fill_value=0).gt(0)
#     if activity.sum(axis=1).gt(1).any():
#         return None
#
#     probabilities = torch.zeros(len(target_classes), dtype=torch.float32)
#     for label in present_classes:
#         probabilities[target_classes.index(label)] = float(activity[label].mean())
#
#     uncovered = float((~activity.any(axis=1)).mean())
#     if len(fallback_classes) == 1:
#         probabilities[target_classes.index(fallback_classes[0])] = uncovered
#     elif len(fallback_classes) > 1 or uncovered > 0:
#         return None
#
#     if not torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-5):
#         return None
#     return probabilities


def build_soft_multiclass_target(target, target_classes: Sequence[str]):
    """Return class probabilities from the temporal coverage in one step."""
    target_classes = list(target_classes)
    if len(target_classes) == 0:
        raise ValueError("target_classes must not be empty.")

    present_classes = [label for label in target_classes if label in target.columns]
    fallback_classes = [label for label in target_classes if label not in target.columns]
    activity = target.reindex(columns=present_classes, fill_value=0).to_numpy(copy=False) > 0
    if np.any(activity.sum(axis=1) > 1):
        return None

    probabilities = np.zeros(len(target_classes), dtype=np.float32)
    for column_idx, label in enumerate(present_classes):
        probabilities[target_classes.index(label)] = activity[:, column_idx].mean()

    uncovered = (~activity.any(axis=1)).mean()
    if len(fallback_classes) == 1:
        probabilities[target_classes.index(fallback_classes[0])] = uncovered
    elif len(fallback_classes) > 1 or uncovered > 0:
        return None

    probabilities = torch.from_numpy(probabilities)
    if not torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-5):
        return None
    return probabilities


# Reference implementation retained temporarily for comparison.
# def build_multiclass_sequence(target, target_classes: Sequence[str], percentage: float, sequence_len: int, soft_boundaries: bool = False):
#     """Split one target window into contiguous categorical sequence steps."""
#     if sequence_len < 1:
#         raise ValueError("sequence_len must be at least 1.")
#     if len(target) % sequence_len != 0:
#         raise ValueError(f"Target length {len(target)} is not divisible by sequence_len={sequence_len}.")
#
#     step_len = len(target) // sequence_len
#     steps = []
#     for step_idx in range(sequence_len):
#         step = target.iloc[step_idx * step_len:(step_idx + 1) * step_len]
#         step_target = build_soft_multiclass_target(step, target_classes) if soft_boundaries else build_multiclass_onehot(step, target_classes=target_classes, percentage=percentage)
#         if step_target is None:
#             return None
#         steps.append(step_target)
#     return torch.stack(steps)


def build_multiclass_sequence(target, target_classes: Sequence[str], percentage: float, sequence_len: int, soft_boundaries: bool = False):
    """Split one target window into contiguous categorical sequence steps."""
    if sequence_len < 1:
        raise ValueError("sequence_len must be at least 1.")
    if len(target) % sequence_len != 0:
        raise ValueError(f"Target length {len(target)} is not divisible by sequence_len={sequence_len}.")

    target_classes = list(target_classes)
    if len(target_classes) == 0:
        raise ValueError("target_classes must not be empty.")

    step_len = len(target) // sequence_len
    present_classes = [label for label in target_classes if label in target.columns]
    fallback_classes = [label for label in target_classes if label not in target.columns]
    values = target.reindex(columns=present_classes, fill_value=0).to_numpy(copy=False).reshape(sequence_len, step_len, len(present_classes))
    sequence_target = np.zeros((sequence_len, len(target_classes)), dtype=np.float32)

    if soft_boundaries:
        activity = values > 0
        if np.any(activity.sum(axis=2) > 1):
            return None

        coverage = activity.mean(axis=1)
        for column_idx, label in enumerate(present_classes):
            sequence_target[:, target_classes.index(label)] = coverage[:, column_idx]

        uncovered = (~activity.any(axis=2)).mean(axis=1)
        if len(fallback_classes) == 1:
            sequence_target[:, target_classes.index(fallback_classes[0])] = uncovered
        elif len(fallback_classes) > 1 or np.any(uncovered > 0):
            return None

        sequence_target = torch.from_numpy(sequence_target)
        if not torch.isclose(sequence_target.sum(dim=1), torch.ones(sequence_len), atol=1e-5).all():
            return None
        return sequence_target

    freq = pd.to_timedelta(target.index.freq).total_seconds()
    threshold = np.float32(step_len * freq * percentage)
    durations = (np.nansum(values, axis=1) * freq).astype(np.float32)
    active = durations >= threshold
    active_count = active.sum(axis=1)
    if np.any(active_count > 1):
        return None

    for column_idx, label in enumerate(present_classes):
        sequence_target[:, target_classes.index(label)] = active[:, column_idx]

    unresolved = active_count == 0
    if np.any(unresolved):
        if len(fallback_classes) != 1:
            return None
        sequence_target[unresolved, target_classes.index(fallback_classes[0])] = 1.0
    return torch.from_numpy(sequence_target)


# Reference implementation retained temporarily for comparison.
# def build_sequence_mask(target, sequence_len: int, step_mask: dict) -> torch.Tensor:
#     """Mark sequence steps whose configured labels cover enough of the step."""
#     unknown = set(step_mask) - {"columns", "percentage"}
#     if unknown:
#         raise ValueError(f"Unknown step_mask fields: {sorted(unknown)}.")
#     columns = list(step_mask.get("columns", []))
#     if len(columns) == 0:
#         raise ValueError("step_mask.columns must not be empty.")
#     percentage = float(step_mask.get("percentage", 0.5))
#     if not 0.0 <= percentage <= 1.0:
#         raise ValueError("step_mask.percentage must be between 0 and 1.")
#     if len(target) % sequence_len != 0:
#         raise ValueError(f"Target length {len(target)} is not divisible by sequence_len={sequence_len}.")
#
#     step_len = len(target) // sequence_len
#     mask = []
#     for step_idx in range(sequence_len):
#         step = target.iloc[step_idx * step_len:(step_idx + 1) * step_len]
#         coverage = float(step.reindex(columns=columns, fill_value=0).any(axis=1).mean())
#         mask.append(coverage >= percentage)
#     return torch.tensor(mask, dtype=torch.bool)


def build_sequence_mask(target, sequence_len: int, step_mask: dict) -> torch.Tensor:
    """Mark sequence steps whose configured labels cover enough of the step."""
    unknown = set(step_mask) - {"columns", "percentage"}
    if unknown:
        raise ValueError(f"Unknown step_mask fields: {sorted(unknown)}.")
    columns = list(step_mask.get("columns", []))
    if len(columns) == 0:
        raise ValueError("step_mask.columns must not be empty.")
    percentage = float(step_mask.get("percentage", 0.5))
    if not 0.0 <= percentage <= 1.0:
        raise ValueError("step_mask.percentage must be between 0 and 1.")
    if len(target) % sequence_len != 0:
        raise ValueError(f"Target length {len(target)} is not divisible by sequence_len={sequence_len}.")

    step_len = len(target) // sequence_len
    values = target.reindex(columns=columns, fill_value=0).to_numpy(copy=False).reshape(sequence_len, step_len, len(columns))
    activity = np.not_equal(values, 0) & pd.notna(values)
    coverage = activity.any(axis=2).mean(axis=1)
    return torch.from_numpy(coverage >= percentage)

def prepare_multiclass_target(
    target,
    target_extra=None,
    patient=None,
    time=None,
    *,
    target_classes: Sequence[str],
    percentage: float = 0.5,
    filters=None,
    sequence_len: int = 1,
    soft_boundaries: bool = False,
    step_mask: dict | None = None,
):
    """Build multiclass sequence targets from a label-activity window.

    Args:
        target: Primary label window as a time-indexed DataFrame with one
            column per label and binary activity values.
        target_extra: Optional second label source for the same window. It is
            transformed independently and does not reject an otherwise valid
            primary target.
        patient: Unused callback argument kept for dataset API compatibility.
        time: Unused callback argument kept for dataset API compatibility.
        target_classes: Output class order for the returned target vectors.
        percentage: Minimum fraction of the window a class must cover to be
            considered active.
        filters: Optional filter specifications applied before target
            construction.
        sequence_len: Number of contiguous categorical targets to construct.
        soft_boundaries: Return probabilities equal to the temporal class
            coverage in each step instead of thresholded one-hot labels.
        step_mask: Optional ``columns`` and ``percentage`` specification used
            to mark which sequence steps contribute to loss and metrics.

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
    if not passes_filters(target, filters):
        return None

    target_values = build_multiclass_sequence(target, target_classes=target_classes, percentage=percentage, sequence_len=sequence_len, soft_boundaries=soft_boundaries)

    if target_values is None:
        return None

    item = {"target": target_values}
    if step_mask is not None:
        target_mask = build_sequence_mask(target, sequence_len=sequence_len, step_mask=step_mask)
        if not target_mask.any():
            return None
        item["target_mask"] = target_mask
    if target_extra is None:
        return item

    target_extra_values = build_multiclass_sequence(
        target_extra,
        target_classes=target_classes,
        percentage=percentage,
        sequence_len=sequence_len,
        soft_boundaries=soft_boundaries,
    )

    if target_extra_values is not None:
        item["target_extra"] = target_extra_values
    return item


def annotation_coverage(target: pd.DataFrame, sequence_len: int, labels: Sequence[str]) -> torch.Tensor:
    """Measure each annotation label's coverage in every output step."""
    if len(target) % sequence_len != 0:
        raise ValueError(f"Annotation length {len(target)} is not divisible by sequence_len={sequence_len}.")
    step_len = len(target) // sequence_len
    return torch.tensor([[float(target.iloc[index * step_len:(index + 1) * step_len].reindex(columns=labels, fill_value=0)[label].mean()) for label in labels] for index in range(sequence_len)], dtype=torch.float32)


def prepare_single_target(target, target_extra=None, patient=None, time=None, *, target_classes: Sequence[str], sequence_len: int, annotation_labels: Sequence[str], percentage: float = 0.5, soft_boundaries: bool = False, filters=None, step_mask=None):
    """Prepare one multiclass target together with annotation coverage."""
    prepared = prepare_multiclass_target(target, target_extra=target_extra, patient=patient, time=time, target_classes=target_classes, sequence_len=sequence_len, percentage=percentage, soft_boundaries=soft_boundaries, filters=filters, step_mask=step_mask)
    if prepared is None:
        return None
    prepared["annotation"] = annotation_coverage(target, sequence_len, annotation_labels)
    return prepared


def normalize_multitask_config(task_config: dict[str, dict]):
    """Validate multitask target settings and derive centered target spans."""
    if len(task_config) == 0:
        raise ValueError("task_config must not be empty.")
    for task, config in task_config.items():
        if "target_resolution" not in config:
            raise ValueError(f"Task '{task}' is missing target_resolution.")
        if "sequence_len" not in config:
            raise ValueError(f"Task '{task}' is missing sequence_len.")

    normalized = OrderedDict()
    used_classes = set()
    for task, config in task_config.items():
        labels = list(config["labels"])
        if len(labels) == 0:
            raise ValueError(f"Task '{task}' does not contain any labels.")
        if "default" not in config:
            raise ValueError(f"Task '{task}' is missing a default entry.")

        default = config["default"]
        percentage = float(config.get("percentage", 0.5))
        target_resolution = pd.to_timedelta(config["target_resolution"])
        if target_resolution <= pd.Timedelta(0):
            raise ValueError(f"Task '{task}' target_resolution must be positive.")
        sequence_len = int(config["sequence_len"])
        if sequence_len < 1:
            raise ValueError(f"Task '{task}' sequence_len must be at least 1.")

        for label in labels:
            if label in used_classes:
                raise ValueError(f"Class '{label}' occurs in multiple tasks.")
            used_classes.add(label)
        if default is not None and default not in labels:
            raise ValueError(f"Default label '{default}' is not part of task '{task}'.")
        if not 0 < percentage <= 1:
            raise ValueError(f"Task '{task}' has invalid percentage '{percentage}'.")

        class_counts = config.get("class_counts")
        if class_counts is not None:
            class_counts = {label: float(count) for label, count in class_counts.items()}
            if set(class_counts) != set(labels):
                raise ValueError(f"Task '{task}' class_counts must contain exactly {labels}, got {sorted(class_counts)}.")
            if any(count <= 0 for count in class_counts.values()):
                raise ValueError(f"Task '{task}' class_counts values must be positive.")

        normalized[task] = {
            "task": task,
            "labels": labels,
            "default": default,
            "percentage": percentage,
            "target_resolution": target_resolution,
            "sequence_len": sequence_len,
            "n_steps": sequence_len,
            "target_span": target_resolution * sequence_len,
            "soft_boundaries": bool(config.get("soft_boundaries", False)),
            "step_mask": dict(config["step_mask"]) if config.get("step_mask") is not None else None,
            "embeddings": config.get("embeddings", "EEG"),
            "n_slices": config.get("n_slices", 50),
            "loss_function": config.get("loss_function"),
            "loss_mode": config.get("loss_mode", "none"),
            "class_weights": dict(config.get("class_weights", {})),
            "class_counts": class_counts,
            "task_weight": float(config.get("task_weight", 1.0)),
        }

    common_target_span = max(config["target_span"] for config in normalized.values())
    for config in normalized.values():
        config["target_offset"] = (common_target_span - config["target_span"]) / 2
    return normalized


def build_multitask_target(targets: pd.DataFrame, task_config: dict[str, dict], raise_error: bool = True):
    """Build padded probability targets and per-step masks for all tasks."""
    if targets is None:
        return None
    task_specs = list(task_config.values())
    max_task_steps = max(config["n_steps"] for config in task_specs)
    max_task_classes = max(len(config["labels"]) for config in task_specs)
    output = torch.zeros((len(task_config), max_task_steps, max_task_classes), dtype=torch.float32)
    output_mask = torch.zeros((len(task_config), max_task_steps), dtype=torch.bool)

    frequency = targets.index.freq or pd.infer_freq(targets.index)
    if frequency is None:
        raise ValueError("Multitask targets require a regular time index.")
    frequency = pd.to_timedelta(frequency)
    common_target_span = max(config["target_span"] for config in task_specs)
    expected_samples = int(round(common_target_span / frequency))
    if len(targets) != expected_samples:
        raise ValueError(f"Expected {expected_samples} target samples for target span {common_target_span}, got {len(targets)}.")

    for task_index, config in enumerate(task_specs):
        task_samples = int(round(config["target_span"] / frequency))
        start = (len(targets) - task_samples) // 2
        task_targets = targets.iloc[start:start + task_samples]
        target_values = task_targets.drop(columns=[config["default"]], errors="ignore") if config["default"] is not None else task_targets
        try:
            values = build_multiclass_sequence(target_values, target_classes=config["labels"], percentage=config["percentage"], sequence_len=config["n_steps"], soft_boundaries=config["soft_boundaries"])
            if values is None:
                raise ValueError("target sequence is ambiguous")
            mask = build_sequence_mask(task_targets, sequence_len=config["n_steps"], step_mask=config["step_mask"]) if config["step_mask"] is not None else torch.ones(config["n_steps"], dtype=torch.bool)
        except ValueError as error:
            if raise_error:
                raise ValueError(f"Could not build target for task '{config['task']}': {error}.") from error
            return None

        output[task_index, :config["n_steps"], :len(config["labels"])] = values
        output_mask[task_index, :config["n_steps"]] = mask

    return output, output_mask


def prepare_multitask_target(target, target_extra=None, patient=None, time=None, class_cnts: Optional[dict[str, dict[str, float]]] = None, task_config: Optional[dict[str, dict]] = None):
    """Prepare one dataset target payload for multitask training."""
    if task_config is None:
        raise ValueError("task_config must not be None.")
    if any("target_span" not in config for config in task_config.values()):
        normalized = normalize_multitask_config(task_config)
        task_config.clear()
        task_config.update(normalized)

    built_target = build_multitask_target(target, task_config, raise_error=False)
    if built_target is None:
        return None
    target_values, target_mask = built_target
    item = {"target": target_values, "target_mask": target_mask}

    if class_cnts:
        keep_probability = 1.0
        for task_index, config in enumerate(task_config.values()):
            task_counts = class_cnts.get(config["task"])
            if task_counts is None:
                continue
            total = sum(float(task_counts.get(label, 0.0)) for label in config["labels"])
            if total <= 0:
                continue
            task_probabilities = {label_index: max(float(task_counts.get(label, 0.0)) / total, 1e-12) for label_index, label in enumerate(config["labels"])}
            minimum_probability = min(task_probabilities.values())
            config_mask = target_mask[task_index, :config["n_steps"]]
            target_indices = target_values[task_index, :config["n_steps"], :len(config["labels"])].argmax(dim=-1)
            for label_index in target_indices[config_mask].tolist():
                keep_probability = min(keep_probability, minimum_probability / task_probabilities[int(label_index)])
        if random.random() > keep_probability:
            return None

    if target_extra is not None:
        extra = build_multitask_target(target_extra, task_config, raise_error=False)
        if extra is not None:
            item["target_extra"], item["target_extra_mask"] = extra
    return item
