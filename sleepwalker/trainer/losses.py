"""Loss and class-count helpers used by Sleepwalker trainers.

This module contains a mix of generic weighting helpers and multitask-specific
mask/count utilities used by `MulticlassTrainer` and `MultiLabelTrainer`.
Tests in `tests/test_multilabel_trainer.py` cover the current conditioning and
class-count behavior.
"""

from typing import Literal, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from sleepwalker.utils import logger


def dice_loss(pred, target, weight = None, epsilon=1e-3):
    """Compute a Dice-style loss for one-hot multiclass targets.

    Args:
        pred: Logits shaped `(B, C, S)`.
        target: One-hot or probability-like targets shaped `(B, C, S)`.
        weight: Optional per-class weights shaped `(C,)`.
        epsilon: Small stabilizer to avoid division by zero.

    Returns:
        A scalar loss tensor.
    """
    pred = torch.softmax(pred, dim=1)

    reduce_dims = (0, *range(2, pred.ndim))
    intersection = (pred * target).sum(dim=reduce_dims)
    union = pred.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)

    dice = (2 * intersection + epsilon) / (union + epsilon)  # (C,)

    if weight is not None:
        dice = dice * weight

    return 1 - dice.mean()

def class_weights_for_loss(user_weights: dict[str, float], class_distribution: dict[str, float], mode:Literal['inverse', 'inverse-log'] = "inverse") -> dict[str, float]:
    """Derive effective class weights from observed class frequencies.

    Args:
        user_weights: Optional manual multipliers keyed by class name.
        class_distribution: Observed class counts keyed by class name.
        mode: Weighting strategy. Current code supports `"inverse"` and
            `"inverse-log"`.

    Returns:
        A dictionary of effective class weights keyed by class name.
    """
    new_weights = {}

    if mode == "inverse":
        # Weight classes by their (inverse) occurrence. This can lead to relatively small losses,
        # hence we will also weight normalize it. This is technically not necessary.
        total_sum = 0
        for c in class_distribution.keys():
            new_weights[c] = user_weights.get(c, 1.0) * np.clip(1.0 / class_distribution[c], min = 1e-4)
            total_sum += new_weights[c]
        
        new_weights = {k:v/total_sum for k,v in new_weights.items()}
    else:
        """
        See 
            - MRASleepNet: a multi-resolution attention network for sleep stage classification using single-channel EEG by Rui Yu, Zhuhuang Zhou, Shuicai Wu, Xiaorong Gao and Guangyu Bin in Journal of Neural Engineering 2022, https://github.com/YuRui8879/MRASleepNet/blob/781aee2d2ff1422c598b099081a4c1d7d4bd05d7/DataAdapter/DataAdapter.py#L110
            - An Attention-Based Deep Learning Approach for Sleep Stage Classification With Single-Channel EEG by Eldele et al. in IEEE TRANSACTIONS ON NEURAL SYSTEMS AND REHABILITATION ENGINEERING 2021, https://github.com/emadeldeen24/AttnSleep/blob/6b4d2665884628c8a7bb09f36589a8ec0992f8e2/utils/util.py#L62
        """

        total = sum(class_distribution.values())
        factor = 1.0 / len(class_distribution)
        for c in class_distribution.keys():
            mu = factor * user_weights.get(c,1.0)
            new_weights[c] = mu * np.clip(np.log( (total * mu) / class_distribution[c]), min=1.0)
        
    return new_weights


def build_multilabel_task_masks(
    y: torch.Tensor,
    target_mask: torch.Tensor,
    task_config: dict[str, dict],
    condition_task: Optional[str] = None,
    condition_labels: Optional[list[str]] = None,
    conditioned_tasks: Optional[list[str]] = None,
):
    """Build per-task step masks for multitask training.

    Args:
        y: Probability targets shaped `[B, n_tasks, max_steps, max_classes]`.
        target_mask: Valid target steps shaped `[B, n_tasks, max_steps]`.
        task_config: Normalized task configuration.
        condition_task: Optional task whose labels gate the conditioned tasks.
        condition_labels: Labels within `condition_task` that activate the
            conditioned tasks.
        conditioned_tasks: Tasks to mask. Defaults to all tasks except the
            condition task.

    Returns:
        A mapping from task name to boolean masks shaped `[B, task_steps]`.

    Raises:
        ValueError: If task names or labels are invalid, or if the condition
            task contains unresolved targets.
    """
    if y.ndim != 4:
        raise ValueError(f"Expected multitask targets [B, T, S, C], got {tuple(y.shape)}.")
    if tuple(target_mask.shape) != tuple(y.shape[:-1]):
        raise ValueError(f"Expected target_mask shaped {tuple(y.shape[:-1])}, got {tuple(target_mask.shape)}.")

    task_specs = list(task_config.values())
    task_masks = {
        cfg["task"]: target_mask[:, task_idx, :cfg["n_steps"]].clone()
        for task_idx, cfg in enumerate(task_specs)
    }
    if condition_task is None:
        return task_masks

    if condition_labels is None or len(condition_labels) == 0:
        raise ValueError("condition_labels must not be empty when condition_task is set.")

    condition_idx = None
    for idx, cfg in enumerate(task_specs):
        if cfg["task"] == condition_task:
            condition_idx = idx
            condition_cfg = cfg
            break
    if condition_idx is None:
        raise ValueError(f"Unknown condition_task '{condition_task}'.")

    unknown_labels = [label for label in condition_labels if label not in condition_cfg["labels"]]
    if len(unknown_labels) > 0:
        raise ValueError(
            f"condition_labels contains labels not present in task '{condition_task}': {unknown_labels}"
        )

    y_condition = y[:, condition_idx, :condition_cfg["n_steps"], :len(condition_cfg["labels"])].argmax(dim=-1)
    condition_mask = torch.zeros_like(y_condition, dtype=torch.bool)
    for idx in [condition_cfg["labels"].index(label) for label in condition_labels]:
        condition_mask |= y_condition == idx
    condition_mask &= task_masks[condition_task]
    condition_mask = condition_mask.any(dim=1, keepdim=True)

    conditioned = set(conditioned_tasks or [task for task in task_config if task != condition_task])
    for task in conditioned:
        if task not in task_config:
            raise ValueError(f"Unknown conditioned task '{task}'.")
        if task == condition_task:
            raise ValueError("condition_task must not also be listed in conditioned_tasks.")
        task_masks[task] &= condition_mask

    return task_masks


def estimate_multilabel_class_cnts(
    loader: DataLoader,
    task_config: dict[str, dict],
    condition_task: Optional[str] = None,
    condition_labels: Optional[list[str]] = None,
    conditioned_tasks: Optional[list[str]] = None,
):
    """Estimate per-task class counts from a multitask dataloader.

    Args:
        loader: Dataloader producing probability targets and step masks.
        task_config: Normalized task configuration.
        condition_task: Optional task used to gate conditioned tasks.
        condition_labels: Activating labels for `condition_task`.
        conditioned_tasks: Tasks whose samples should be masked by the
            condition task.

    Returns:
        Nested dictionaries of class counts keyed as `task -> label -> count`.

    Notes:
        Tests confirm that the current implementation respects conditioning
        masks while counting conditioned tasks.
    """
    class_cnts = {
        cfg["task"]: torch.zeros(len(cfg["labels"]), dtype=torch.float64)
        for cfg in task_config.values()
    }
    batch_size = loader.batch_size or 1
    logger.progress_start(len(loader) * batch_size, desc="Estimating class counts", leave=True)
    for batch in loader:
        if batch is None:
            continue
        y = batch["target"]
        if "target_mask" not in batch:
            raise ValueError("Multitask batches must contain target_mask.")
        target_mask = batch["target_mask"].to(dtype=torch.bool)
        task_masks = build_multilabel_task_masks(
            y,
            target_mask,
            task_config,
            condition_task=condition_task,
            condition_labels=condition_labels,
            conditioned_tasks=conditioned_tasks,
        )
        for task_idx, cfg in enumerate(task_config.values()):
            y_task = y[:, task_idx, :cfg["n_steps"], :len(cfg["labels"])]
            y_selected = y_task[task_masks[cfg["task"]]]
            if y_selected.numel() == 0:
                continue
            class_cnts[cfg["task"]] += y_selected.sum(dim=0).to(dtype=torch.float64)
        logger.progress_advance(y.shape[0])
    logger.progress_close()

    return {
        task: {label: float(class_cnts[task][idx].item()) for idx, label in enumerate(task_config[task]["labels"])}
        for task in task_config
    }
