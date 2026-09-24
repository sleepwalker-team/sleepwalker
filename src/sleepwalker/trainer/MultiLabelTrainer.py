"""Concrete trainer for multi-task, multi-resolution classification.

The repository uses this trainer for experiments where one input window
produces several task-specific categorical predictions, potentially at
different temporal resolutions.
"""

from functools import partial
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from sleepwalker.trainer.BaseTrainer import BaseTrainer
from sleepwalker.trainer.losses import build_multilabel_task_masks, class_weights_for_loss, estimate_multilabel_class_cnts
from sleepwalker.trainer.utils.display import format_confusion_table, render_confusion_table_grid
from sleepwalker.metrics import accuracy_from_confusion_matrix, cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix
from sleepwalker.trainer.utils.targets import normalize_multitask_config
from sleepwalker.utils import logger


class MultiLabelTrainer(BaseTrainer):
    """
    Train a classifier with one categorical head per configured task.

    Each task contributes its own label set and target resolution. Smaller
    target resolutions are expanded into multiple steps inside the largest
    dataset target window.

    Args:
        epochs: Number of training epochs.
        optimizer: Factory that builds an optimizer for the model.
        task_config: Task specification mapping task names to label sets,
            target resolutions, sequence lengths, and loss settings.
        condition_task: Optional task whose labels gate the loss for other
            tasks.
        condition_labels: Labels within ``condition_task`` that activate the
            conditioned tasks.
        conditioned_tasks: Tasks whose losses should be masked by
            ``condition_task``.
        device: Torch device used for training and inference.
        warmup_device: Device used during preprocessor warmup.
        save_every: Checkpoint cadence in epochs.
        eval_every: Validation cadence in epochs.
        lr_scheduler: Optional scheduler factory.
        early_stopping: Optional validation patience.
        return_best: Whether to return the lowest-loss validation checkpoint
            instead of leaving the model at its last training epoch.
        log_batches: Whether to emit batch-level metric logs.

    """

    def __init__(
        self,
        epochs: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        task_config: dict[str, dict],
        condition_task: Optional[str] = None,
        condition_labels: Optional[list[str]] = None,
        conditioned_tasks: Optional[list[str]] = None,
        device: str = "cuda:0",
        warmup_device: str = "cpu",
        save_every: int = 10,
        eval_every: int = 10,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
        early_stopping: Optional[int] = None,
        return_best: bool = True,
        log_batches: bool = True,
    ):
        super().__init__(
            epochs=epochs,
            optimizer=optimizer,
            device=device,
            warmup_device=warmup_device,
            save_every=save_every,
            eval_every=eval_every,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            return_best=return_best,
        )
        self.log_batches = log_batches

        self.task_config = normalize_multitask_config(task_config)
        self.task_specs = list(self.task_config.values())
        self.task_idx = {cfg["task"]: idx for idx, cfg in enumerate(self.task_specs)}
        self.target_resolution = max(cfg["target_span"] for cfg in self.task_specs)
        self.max_task_steps = max(cfg["n_steps"] for cfg in self.task_specs)
        self.max_task_classes = max(len(cfg["labels"]) for cfg in self.task_specs)
        self.classes = [label for cfg in self.task_config.values() for label in cfg["labels"]]
        self.num_classes = len(self.classes)
        self.num_tasks = len(self.task_specs)

        if condition_task is None:
            if condition_labels is not None or conditioned_tasks is not None:
                raise ValueError("condition_labels and conditioned_tasks require condition_task to be set.")
            self.condition_task = None
            self.condition_task_idx = None
            self.condition_label_idx = set()
            self.conditioned_tasks = set()
        else:
            if condition_task not in self.task_config:
                raise ValueError(f"Unknown condition_task '{condition_task}'.")
            if condition_labels is None or len(condition_labels) == 0:
                raise ValueError("condition_labels must not be empty when condition_task is set.")

            condition_cfg = self.task_config[condition_task]
            unknown_labels = [label for label in condition_labels if label not in condition_cfg["labels"]]
            if len(unknown_labels) > 0:
                raise ValueError(
                    f"condition_labels contains labels not present in task '{condition_task}': {unknown_labels}"
                )

            if conditioned_tasks is None:
                conditioned_tasks = [task for task in self.task_config if task != condition_task]
            unknown_tasks = [task for task in conditioned_tasks if task not in self.task_config]
            if len(unknown_tasks) > 0:
                raise ValueError(f"conditioned_tasks contains unknown tasks: {unknown_tasks}")
            if condition_task in conditioned_tasks:
                raise ValueError("condition_task must not also be listed in conditioned_tasks.")

            self.condition_task = condition_task
            self.condition_task_idx = self.task_idx[condition_task]
            self.condition_label_idx = {condition_cfg["labels"].index(label) for label in condition_labels}
            self.conditioned_tasks = set(conditioned_tasks)

        self.task_loss_functions = {}
        self.task_loss_weights = {}
        for cfg in self.task_specs:
            task = cfg["task"]
            if cfg.get("loss_function") is None:
                raise ValueError(f"Task '{task}' is missing loss_function.")
            self.task_loss_functions[task] = cfg["loss_function"]
            self.task_loss_weights[task] = float(cfg["task_weight"])

    def classification_contract(self) -> dict:
        return {
            "type": "multitask",
            "tasks": {
                task: {
                    "classes": list(config["labels"]),
                    "n_steps": int(config["n_steps"]),
                    "target_resolution": str(config["target_resolution"]),
                    "target_offset": str(config["target_offset"]),
                    "default": config["default"],
                    "percentage": float(config["percentage"]),
                    "soft_boundaries": bool(config["soft_boundaries"]),
                }
                for task, config in self.task_config.items()
            },
        }

    def warmup_trainer(self, data_loader, device: str = "cuda"):
        """Estimate per-task class counts and configure weighted losses."""
        tasks_needing_counts = [
            cfg["task"]
            for cfg in self.task_config.values()
            if (cfg.get("loss_mode", "none") or "none") != "none"
        ]
        if not tasks_needing_counts:
            return

        class_cnts = {
            task: dict(self.task_config[task]["class_counts"])
            for task in tasks_needing_counts
            if self.task_config[task]["class_counts"] is not None
        }
        missing_tasks = [task for task in tasks_needing_counts if task not in class_cnts]
        if missing_tasks:
            estimated = estimate_multilabel_class_cnts(
                data_loader,
                self.task_config,
                condition_task=self.condition_task,
                condition_labels=[
                    self.task_config[self.condition_task]["labels"][idx]
                    for idx in sorted(self.condition_label_idx)
                ] if self.condition_task is not None else None,
                conditioned_tasks=list(self.conditioned_tasks) if self.condition_task is not None else None,
            )
            class_cnts.update({task: estimated[task] for task in missing_tasks})
            logger.info(f"Estimated class counts for {missing_tasks}: {class_cnts}")
        else:
            logger.info(f"Using configured class counts: {class_cnts}")

        for cfg in self.task_specs:
            task = cfg["task"]
            loss_function = cfg.get("loss_function") or torch.nn.functional.cross_entropy
            loss_mode = cfg.get("loss_mode", "none") or "none"
            class_weights = dict(cfg.get("class_weights", {}))
            if loss_mode not in {"none", "inverse", "inverse-log"}:
                raise ValueError(f"Task '{task}' has invalid loss_mode '{loss_mode}'.")
            if loss_mode != "none":
                task_cnts = class_cnts.get(task)
                if task_cnts is None:
                    raise ValueError(f"Missing class counts for task '{task}'.")
                task_cnts = {label: max(float(task_cnts.get(label, 0.0)), 1.0) for label in cfg["labels"]}
                class_weights = class_weights_for_loss(class_weights, task_cnts, loss_mode)

            if len(class_weights) > 0:
                weights_torch = torch.tensor(
                    [class_weights.get(label, 1.0) for label in cfg["labels"]],
                    device=self.device,
                    dtype=torch.float32,
                )
                self.task_loss_functions[task] = partial(loss_function, weight=weights_torch)
            else:
                self.task_loss_functions[task] = loss_function

    def _log_from_cms(self, cms: dict[str, np.ndarray], loss_value: float, mode: str, scope: str = "batch", step: int = 0):
        metrics = []
        for cm in cms.values():
            total = cm.sum()
            if total == 0:
                continue

            metrics.append({
                "accuracy": accuracy_from_confusion_matrix(cm),
                "f1_micro": f1_score_from_confusion_matrix(cm, macro=False),
                "f1_macro": f1_score_from_confusion_matrix(cm, macro=True),
                "kappa": cohen_kappa_from_confusion_matrix(cm),
            })

        if len(metrics) == 0:
            return

        logger.metric(f"{scope}/{mode}/accuracy", np.mean([m["accuracy"] for m in metrics]), step=step)
        logger.metric(f"{scope}/{mode}/f1_micro", np.mean([m["f1_micro"] for m in metrics]), step=step)
        logger.metric(f"{scope}/{mode}/f1_macro", np.mean([m["f1_macro"] for m in metrics]), step=step)
        logger.metric(f"{scope}/{mode}/coehns_kappa", np.mean([m["kappa"] for m in metrics]), step=step)
        logger.metric(f"{scope}/{mode}/loss", float(loss_value), step=step)

    # TODO REFACTOR
    def format_confusion_table(self, task: str, cm: np.ndarray) -> list[str]:
        return format_confusion_table(list(self.task_config[task]["labels"]), cm, title=task)

    def log_confusion_tables(self, mode: str, cm_sum: dict[str, np.ndarray]):
        blocks = [self.format_confusion_table(task, cm) for task, cm in cm_sum.items()]
        if len(blocks) > 0:
            logger.info(render_confusion_table_grid(blocks, header=f"{mode.upper()} confusion matrices"))

    def run_epoch(self, loader, opt, model, prefix="", lr_scheduler=None):
        """Run one train, validation, or test epoch for a multitask model.

        Args:
            loader: Dataloader producing ``data``, probability targets, and
                per-task step masks.
            opt: Optimizer for training epochs, or ``None`` during evaluation.
            model: Expected to expose ``task_config`` and return task logits.
            prefix: Progress-label prefix used to infer the logging mode.

        Returns:
            A tuple ``(epoch_loss, task_confusion_matrices)`` where the second
            element is a mapping from task name to NumPy confusion matrix.

        Raises:
            ValueError: If the model type, dataset resolution, logits, or
                targets do not match the configured task layout.
        """
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)

        if not hasattr(self, "steps"):
            self.steps = {"train": 0, "val": 0, "test": 0}
        if not hasattr(self, "epoch_step"):
            self.epoch_step = 0

        loss_sum = 0.0
        loss_weight_sum = 0
        epoch_cms = {
            cfg["task"]: torch.zeros((len(cfg["labels"]), len(cfg["labels"])), dtype=torch.int64, device=self.device)
            for cfg in self.task_specs
        }
        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"
        per_task_losses = [0.0 for _ in range(len(self.task_specs))]
        per_task_loss_weights = [0 for _ in range(len(self.task_specs))]
        per_task_correct = [0 for _ in range(len(self.task_specs))]
        per_task_counts = [0 for _ in range(len(self.task_specs))]
        for batch in loader:
            if batch is None:
                continue
            if opt is not None:
                opt.zero_grad(set_to_none=True)

            x = {key: value.to(self.device, non_blocking=True) for key, value in batch["data"].items()} if isinstance(batch["data"], dict) else batch["data"].to(self.device, non_blocking=True)
            logits = model(x)
            y = batch["target"].to(self.device, non_blocking=True)
            if "target_mask" not in batch:
                raise ValueError("Multitask batches must contain target_mask.")
            target_mask = batch["target_mask"].to(self.device, dtype=torch.bool, non_blocking=True)

            task_masks = build_multilabel_task_masks(
                y,
                target_mask,
                self.task_config,
                condition_task=self.condition_task,
                condition_labels=[
                    self.task_config[self.condition_task]["labels"][idx]
                    for idx in sorted(self.condition_label_idx)
                ] if self.condition_task is not None else None,
                conditioned_tasks=list(self.conditioned_tasks) if self.condition_task is not None else None,
            )

            losses = []
            batch_cms = {} if self.log_batches else None
            for task_idx, cfg in enumerate(self.task_specs):
                task = cfg["task"]
                n_classes = len(cfg["labels"])
                n_steps = cfg["n_steps"]
                y_task = y[:, task_idx, :n_steps, :n_classes]
                logits_task = logits[task]
                step_mask = task_masks[task]

                expected_shape = (y.shape[0], n_steps, n_classes)
                if tuple(logits_task.shape) != expected_shape:
                    raise ValueError(
                        f"Task '{task}' logits shape {tuple(logits_task.shape)} does not match expected {expected_shape}."
                    )
                if tuple(y_task.shape) != expected_shape:
                    raise ValueError(f"Task '{task}' target shape {tuple(y_task.shape)} does not match expected {expected_shape}.")
                if tuple(step_mask.shape) != expected_shape[:2]:
                    raise ValueError(f"Task '{task}' mask shape {tuple(step_mask.shape)} does not match expected {expected_shape[:2]}.")
                if not step_mask.any():
                    continue

                logits_selected = logits_task[step_mask]
                y_selected = y_task[step_mask]
                if not torch.allclose(y_selected.sum(dim=-1), torch.ones(y_selected.shape[0], device=y_selected.device), atol=1e-5):
                    raise ValueError(f"Task '{task}' valid targets must sum to one.")

                loss_task = self.task_loss_functions[task](logits_selected, y_selected)
                loss_task = self.task_loss_weights[task] * loss_task
                pred_selected = logits_selected.argmax(dim=-1)
                pred_flat = pred_selected.reshape(-1)
                y_flat = y_selected.argmax(dim=-1).reshape(-1)

                task_loss_weight = int(y_flat.numel())
                per_task_losses[task_idx] += float(loss_task.item()) * task_loss_weight
                per_task_loss_weights[task_idx] += task_loss_weight
                per_task_correct[task_idx] += int((pred_flat == y_flat).sum().item())
                per_task_counts[task_idx] += int(y_flat.numel())
                losses.append(loss_task)

                counts = torch.bincount(
                    y_flat.to(dtype=torch.int64) * n_classes + pred_flat.to(dtype=torch.int64),
                    minlength=n_classes * n_classes,
                ).reshape(n_classes, n_classes)
                epoch_cms[task] += counts
                if self.log_batches and batch_cms is not None:
                    batch_cms[task] = counts.detach().cpu().numpy()

            if len(losses) == 0:
                continue
            loss = torch.stack(losses).mean()

            if opt is not None:
                loss.backward()
                opt.step()

            batch_loss_weight = int(y.shape[0])
            loss_sum += float(loss.item()) * batch_loss_weight
            loss_weight_sum += batch_loss_weight

            if self.log_batches and batch_cms:
                self._log_from_cms(batch_cms, float(loss.item()), mode=mode, scope="batch", step=self.steps[mode])

            task_desc_parts = []
            for task_idx, task in enumerate(self.task_specs):
                if per_task_loss_weights[task_idx] == 0 or per_task_counts[task_idx] == 0:
                    task_desc_parts.append(f"{task['task'][:5]}: n/a")
                    continue
                task_loss = per_task_losses[task_idx] / per_task_loss_weights[task_idx]
                task_acc = per_task_correct[task_idx] / per_task_counts[task_idx] * 100.0
                task_desc_parts.append(f"{task['task'][:5]}: {task_loss:2.4f} - {task_acc:2.2f}")
            task_desc = " ".join(task_desc_parts)
            desc = f"{prefix:<12} loss {loss_sum/loss_weight_sum:2.4f} {task_desc}"

            self.steps[mode] += 1
            logger.progress_status(desc)
            logger.progress_advance(loader.batch_size)
            if lr_scheduler is not None:
                lr_scheduler.step()

        logger.progress_close()
        epoch_loss = loss_sum / max(loss_weight_sum, 1)
        epoch_cms_np = {
            task: cm.detach().cpu().numpy()
            for task, cm in epoch_cms.items()
        }
        self._log_from_cms(epoch_cms_np, epoch_loss, mode=mode, scope="epoch", step=self.epoch_step)
        self.log_confusion_tables(mode, epoch_cms_np)

        return epoch_loss, epoch_cms_np
