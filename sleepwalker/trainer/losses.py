from typing import Literal, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from sleepwalker.utils import logger


def dice_loss(pred, target, weight = None, epsilon=1e-3):
    """
    Dice loss for multi-class targets.
    Args:
        pred (torch.Tensor): Logits of shape (B, C).
        target (torch.Tensor): Class indices of shape (B,).
        weight (torch.Tensor): Class weights of shape (C,).
        epsilon (float): Small constant to avoid division by zero.
    Returns:
        torch.Tensor: Scalar Dice loss.
    """
    pred = torch.softmax(pred, dim=1)  # (B, C)
    #target_onehot = torch.nn.functional.one_hot(target.long(), num_classes=pred.shape[1]).float()  # (B, C)

    intersection = (pred * target).sum(dim=0)
    union = pred.sum(dim=0) + target.sum(dim=0)

    dice = (2 * intersection + epsilon) / (union + epsilon)  # (C,)

    if weight is not None:
        dice = dice * weight

    return 1 - dice.mean()

def class_weights_for_loss(user_weights: dict[str, float], class_distribution: dict[str, float], mode:Literal['inverse', 'inverse-log'] = "inverse") -> dict[str, float]:
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
    task_config: dict[str, dict],
    condition_task: Optional[str] = None,
    condition_labels: Optional[list[str]] = None,
    conditioned_tasks: Optional[list[str]] = None,
):
    task_specs = list(task_config.values())
    task_masks = {
        cfg["task"]: torch.ones(y.shape[0], dtype=torch.bool, device=y.device)
        for cfg in task_specs
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

    y_condition = y[:, condition_idx, :condition_cfg["n_steps"]]
    if (y_condition < 0).any():
        raise ValueError(
            f"Condition task '{condition_task}' contains invalid targets. "
            "Unclear labels must be filtered in get_target()."
        )

    condition_mask = torch.zeros_like(y_condition, dtype=torch.bool)
    for idx in [condition_cfg["labels"].index(label) for label in condition_labels]:
        condition_mask |= y_condition == idx
    condition_mask = condition_mask.any(dim=1)

    conditioned = set(conditioned_tasks or [task for task in task_config if task != condition_task])
    for task in conditioned:
        if task not in task_config:
            raise ValueError(f"Unknown conditioned task '{task}'.")
        if task == condition_task:
            raise ValueError("condition_task must not also be listed in conditioned_tasks.")
        task_masks[task] = condition_mask

    return task_masks


def estimate_multilabel_class_cnts(
    loader: DataLoader,
    task_config: dict[str, dict],
    condition_task: Optional[str] = None,
    condition_labels: Optional[list[str]] = None,
    conditioned_tasks: Optional[list[str]] = None,
):
    class_cnts = {
        cfg["task"]: torch.zeros(len(cfg["labels"]), dtype=torch.float64)
        for cfg in task_config.values()
    }
    batch_size = loader.batch_size or 1
    logger.progress_start(len(loader) * batch_size, desc="Estimating class counts", leave=True)
    for batch in loader:
        y = batch["target"]
        task_masks = build_multilabel_task_masks(
            y,
            task_config,
            condition_task=condition_task,
            condition_labels=condition_labels,
            conditioned_tasks=conditioned_tasks,
        )
        for task_idx, cfg in enumerate(task_config.values()):
            y_task = y[:, task_idx, :cfg["n_steps"]]
            y_selected = y_task[task_masks[cfg["task"]]]
            if y_selected.numel() == 0:
                continue
            if (y_selected < 0).any():
                raise ValueError(f"Task '{cfg['task']}' contains invalid targets while estimating class counts.")
            counts = torch.bincount(y_selected.reshape(-1), minlength=len(cfg["labels"]))
            class_cnts[cfg["task"]] += counts.to(dtype=torch.float64)
        logger.progress_advance(y.shape[0])
    logger.progress_close()

    return {
        task: {label: float(class_cnts[task][idx].item()) for idx, label in enumerate(task_config[task]["labels"])}
        for task in task_config
    }
