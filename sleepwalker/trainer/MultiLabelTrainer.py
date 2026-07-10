"""Concrete trainer for multi-task, multi-resolution classification.

The repository uses this trainer for experiments where one input window
produces several task-specific categorical predictions, potentially at
different temporal resolutions. The canonical example in the current tree is
``train_multilabel.py`` together with ``MetaModel``.
"""

from functools import partial
import random
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from sleepwalker.datasets.utils import RepeatSampler
from sleepwalker.models.MetaModel import MetaModel
from sleepwalker.trainer.BaseTrainer import BaseTrainer
from sleepwalker.trainer.losses import build_multilabel_task_masks, class_weights_for_loss, estimate_multilabel_class_cnts
from sleepwalker.trainer.utils.display import format_confusion_table, render_confusion_table_grid
from sleepwalker.trainer.utils.metrics import cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix
from sleepwalker.trainer.utils.targets import resolve_multiclass_index
from sleepwalker.utils import logger


class MultiLabelTrainer(BaseTrainer):
    """
    Train a ``MetaModel`` with one categorical head per configured task.

    Each task contributes its own label set and target resolution. Smaller
    target resolutions are expanded into multiple steps inside the largest
    dataset target window.

    Args:
        epochs: Number of training epochs.
        optimizer: Factory that builds an optimizer for the model.
        task_config: Normalized task specification mapping task names to label
            sets, target resolutions, and loss settings.
        condition_task: Optional task whose labels gate the loss for other
            tasks.
        condition_labels: Labels within ``condition_task`` that activate the
            conditioned tasks.
        conditioned_tasks: Tasks whose losses should be masked by
            ``condition_task``.
        device: Torch device used for training and inference.
        warmup_device: Device used during preprocessor warmup.
        save_every: Checkpoint cadence in epochs.
        lr_scheduler: Optional scheduler factory.
        early_stopping: Optional validation patience.
        log_batches: Whether to emit batch-level metric logs.
        n_repeat_train: Number of repeated views per training sample.
        n_repeat_test: Number of repeated views per evaluation sample.

    Notes:
        ``task_config`` must already be normalized before construction. The
        repository typically does this with :meth:`normalize_task_config`.
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
        save_every: int = 1,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
        early_stopping: Optional[int] = None,
        log_batches: bool = True,
        n_repeat_train: int = 1,
        n_repeat_test: int = 1,
    ):
        super().__init__(
            epochs=epochs,
            optimizer=optimizer,
            device=device,
            warmup_device=warmup_device,
            save_every=save_every,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            n_repeat_train=n_repeat_train,
            n_repeat_test=n_repeat_test,
        )
        self.log_batches = log_batches

        self.task_config = OrderedDict(task_config)
        self.task_specs = list(self.task_config.values())
        self.task_idx = {cfg["task"]: idx for idx, cfg in enumerate(self.task_specs)}
        self.largest_task_resolution = max(cfg["target_resolution"] for cfg in self.task_specs)
        self.max_task_steps = max(cfg["n_steps"] for cfg in self.task_specs)
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

            condition_resolution = condition_cfg["target_resolution"]
            for task in conditioned_tasks:
                task_resolution = self.task_config[task]["target_resolution"]
                ratio = condition_resolution / task_resolution
                if ratio != int(ratio):
                    raise ValueError(
                        f"Task '{task}' target_resolution={task_resolution} must divide "
                        f"condition_task '{condition_task}' target_resolution={condition_resolution}."
                    )

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

    def warmup_trainer(self, data_loader, device: str = "cuda"):
        """Estimate per-task class counts and configure weighted losses."""
        needs_counts = any((cfg.get("loss_mode", "none") or "none") != "none" for cfg in self.task_config.values())
        if not needs_counts:
            return

        class_cnts = estimate_multilabel_class_cnts(
            data_loader,
            self.task_config,
            condition_task=self.condition_task,
            condition_labels=[
                self.task_config[self.condition_task]["labels"][idx]
                for idx in sorted(self.condition_label_idx)
            ] if self.condition_task is not None else None,
            conditioned_tasks=list(self.conditioned_tasks) if self.condition_task is not None else None,
        )
        logger.info(f"Estimated class counts: {class_cnts}")

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

    def _prediction_frame(self, batch, outputs) -> pd.DataFrame:
        """Convert multitask logits into the repository's wide prediction table."""
        rows = []
        batch_size = len(batch.get("time", []))
        largest_resolution = pd.to_timedelta(self.largest_task_resolution)
        for sample_idx in range(batch_size):
            row = {
                "patient": batch["patient"][sample_idx],
                "time": batch["time"][sample_idx],
            }
            largest_start = pd.Timestamp(batch["time"][sample_idx]) - largest_resolution / 2
            for task, cfg in self.task_config.items():
                task_logits = outputs[task][sample_idx].detach().cpu()
                task_probs = torch.softmax(task_logits, dim=-1)
                task_pred_idx = task_probs.argmax(dim=-1)
                task_resolution = pd.to_timedelta(cfg["target_resolution"])
                for step_idx in range(cfg["n_steps"]):
                    suffix = "" if cfg["n_steps"] == 1 else f"__step_{step_idx}"
                    step_center = largest_start + step_idx * task_resolution + task_resolution / 2
                    pred_idx = int(task_pred_idx[step_idx].item())
                    row[f"{task}{suffix}__time"] = step_center
                    row[f"{task}{suffix}__prediction_idx"] = pred_idx
                    row[f"{task}{suffix}__prediction"] = cfg["labels"][pred_idx]
                    for label_idx, label in enumerate(cfg["labels"]):
                        row[f"{task}{suffix}__prob__{label}"] = float(task_probs[step_idx, label_idx].item())
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def normalize_task_config(task_config: dict[str, dict]):
        """Validate and normalize a raw multitask configuration.

        Args:
            task_config: User-provided task configuration keyed by task name.

        Returns:
            An ``OrderedDict`` whose values include normalized pandas
            timedeltas, inferred ``n_steps``, and defaulted loss settings.

        Raises:
            ValueError: If labels are duplicated across tasks, defaults are
                invalid, or task resolutions are incompatible.
        """
        normalized = OrderedDict()
        used_classes = set()
        for task, cfg in task_config.items():
            if "target_resolution" not in cfg:
                raise ValueError(f"Task '{task}' is missing target_resolution.")
        largest_task_resolution = max(pd.to_timedelta(cfg["target_resolution"]) for cfg in task_config.values())

        for task, cfg in task_config.items():
            labels = list(cfg["labels"])
            if len(labels) == 0:
                raise ValueError(f"Task '{task}' does not contain any labels.")
            if "default" not in cfg:
                raise ValueError(f"Task '{task}' is missing a default entry.")

            default = cfg["default"]
            percentage = cfg.get("percentage", 0.5)
            target_resolution = pd.to_timedelta(cfg["target_resolution"])

            for label in labels:
                if label in used_classes:
                    raise ValueError(f"Class '{label}' occurs in multiple tasks.")
                used_classes.add(label)

            if default is not None and default not in labels:
                raise ValueError(f"Default label '{default}' is not part of task '{task}'.")
            if percentage <= 0 or percentage > 1:
                raise ValueError(f"Task '{task}' has invalid percentage '{percentage}'.")
            ratio = largest_task_resolution / target_resolution
            if ratio != int(ratio):
                raise ValueError(
                    f"Task '{task}' target_resolution={target_resolution} must divide largest_task_resolution={largest_task_resolution}."
                )

            spec = {
                "task": task,
                "labels": labels,
                "default": default,
                "percentage": percentage,
                "target_resolution": target_resolution,
                "n_steps": int(ratio),
                "embeddings": cfg.get("embeddings", "EEG"),
                "n_slices": cfg.get("n_slices", 50),
                "loss_function": cfg.get("loss_function"),
                "loss_mode": cfg.get("loss_mode", "none"),
                "class_weights": dict(cfg.get("class_weights", {})),
                "task_weight": float(cfg.get("task_weight", 1.0)),
            }
            normalized[task] = spec

        return normalized

    @staticmethod
    def build_multitask_target(targets: pd.DataFrame, task_config: dict[str, dict], raise_error=True):
        """Convert label indicator timelines into integer task targets.

        Args:
            targets: Time-indexed indicator DataFrame for one dataset target
                window.
            task_config: Normalized task configuration.
            raise_error: Whether ambiguous or invalid windows should raise
                ``ValueError`` instead of returning ``None``.

        Returns:
            A tensor shaped ``[n_tasks, max_task_steps]`` containing per-step
            class indices, with ``-1`` reserved for unresolved steps.
        """
        task_specs = list(task_config.values())
        max_task_steps = max(cfg["n_steps"] for cfg in task_specs)
        out = torch.full((len(task_config), max_task_steps), -1, dtype=torch.long)
        freq_seconds = pd.to_timedelta(targets.index.freq).total_seconds()

        for task_idx, cfg in enumerate(task_specs):
            step_seconds = cfg["target_resolution"].total_seconds()
            step_len = int(round(step_seconds / freq_seconds))
            if step_len <= 0:
                raise ValueError(f"Task '{cfg['task']}' has invalid step length.")
            if step_len * cfg["n_steps"] != len(targets):
                raise ValueError(
                    f"Task '{cfg['task']}' expects {cfg['n_steps']} steps of {step_len} samples, got {len(targets)} target samples."
                )

            for step_idx in range(cfg["n_steps"]):
                step_df = targets.iloc[step_idx * step_len:(step_idx + 1) * step_len]
                totals = {str(k): float(v) * freq_seconds for k, v in step_df.sum().to_dict().items()}
                try:
                    labels = cfg["labels"]
                    default = cfg["default"]
                    min_event_seconds = step_seconds * cfg.get("percentage", 0.5)
                    target = torch.tensor([float(totals.get(label, 0.0)) for label in labels], dtype=torch.float)
                    default_idx = labels.index(default) if default is not None else None
                    idx = resolve_multiclass_index(target, default_idx, min_event_seconds, raise_error=raise_error)
                    step_target = -1 if idx is None else idx
                    if step_target < 0 and not raise_error:
                        return None
                    out[task_idx, step_idx] = step_target
                except ValueError as exc:
                    if raise_error:
                        raise ValueError(f"{exc} Task '{cfg['task']}', step {step_idx}.") from exc
                    return None

        return out

    @staticmethod
    def prepare_target(
        target,
        target_extra=None,
        patient=None,
        time=None,
        class_cnts: Optional[dict[str, dict[str, float]]] = None,
        task_config: Optional[dict[str, dict]] = None,
    ):
        """Prepare one dataset target payload for multitask training.

        Args:
            target: Primary target indicator DataFrame.
            target_extra: Optional secondary target indicator DataFrame.
            patient: Patient identifier, currently passed through for callback
                compatibility.
            time: Window timestamp, currently passed through for callback
                compatibility.
            class_cnts: Optional per-task class counts used for rejection-based
                balancing.
            task_config: Normalized task configuration.

        Returns:
            A dictionary containing ``target`` and optionally ``target_extra``,
            or ``None`` when the window should be rejected.
        """
        if task_config is None:
            raise ValueError("task_config must not be None.")

        target = MultiLabelTrainer.build_multitask_target(target, task_config, raise_error=False)
        if target is None:
            return None

        item = {"target": target}

        if class_cnts:
            keep_probability = 1.0
            for task_idx, cfg in enumerate(task_config.values()):
                task = cfg["task"]
                task_cnts = class_cnts.get(task)
                if task_cnts is None:
                    continue

                total = sum(float(task_cnts.get(label, 0.0)) for label in cfg["labels"])
                if total <= 0:
                    continue

                task_probas = {
                    label_idx: max(float(task_cnts.get(label, 0.0)) / total, 1e-12)
                    for label_idx, label in enumerate(cfg["labels"])
                }
                min_proba = min(task_probas.values())
                for label_idx in target[task_idx].tolist():
                    if label_idx < 0:
                        continue
                    keep_probability = min(keep_probability, min_proba / task_probas[int(label_idx)])

            if random.random() > keep_probability:
                return None

        if target_extra is not None:
            extra = MultiLabelTrainer.build_multitask_target(target_extra, task_config, raise_error=False)
            if extra is not None:
                item["target_extra"] = extra

        return item

    def _log_from_cms(self, cms: dict[str, np.ndarray], loss_value: float, mode: str, scope: str = "batch", step: int = 0):
        metrics = []
        for cm in cms.values():
            total = cm.sum()
            if total == 0:
                continue

            metrics.append({
                "accuracy": cm.trace() / total * 100.0,
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

    def run_epoch(self, loader, opt, model, prefix=""):
        """Run one train, validation, or test epoch for a multitask model.

        Args:
            loader: Dataloader producing ``data`` and integer multitask targets.
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
        if not isinstance(model, MetaModel) and not hasattr(model, "task_config"):
            raise ValueError("MultiLabelTrainer requires a multitask model with task_config.")
        dataset_resolution = getattr(loader.dataset, "target_resolution", None)
        if dataset_resolution is None:
            raise ValueError("MultiLabelTrainer requires loader.dataset.target_resolution.")
        dataset_resolution = pd.to_timedelta(dataset_resolution)
        if dataset_resolution != self.largest_task_resolution:
            raise ValueError(
                f"Dataset target_resolution={dataset_resolution} does not match largest_task_resolution={self.largest_task_resolution}."
            )

        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)

        if not hasattr(self, "steps"):
            self.steps = {"train": 0, "val": 0, "test": 0}
        if not hasattr(self, "epoch_step"):
            self.epoch_step = 0

        loss_sum = 0.0
        epoch_cms = {
            cfg["task"]: torch.zeros((len(cfg["labels"]), len(cfg["labels"])), dtype=torch.int64, device=self.device)
            for cfg in self.task_specs
        }
        cnt = 0
        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"
        n_repeat = loader.sampler.n_repeat if isinstance(getattr(loader, "sampler", None), RepeatSampler) else 1

        per_task_losses = [0.0 for _ in range(len(self.task_specs))]
        per_task_loss_batches = [0 for _ in range(len(self.task_specs))]
        per_task_correct = [0 for _ in range(len(self.task_specs))]
        per_task_counts = [0 for _ in range(len(self.task_specs))]
        for batch in loader:
            x = batch["data"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)

            if opt is not None:
                opt.zero_grad(set_to_none=True)

            if n_repeat > 1:
                if x.shape[0] % n_repeat != 0:
                    raise ValueError(f"Batch size {x.shape[0]} is not divisible by n_repeat={n_repeat}.")
                base_batch = x.shape[0] // n_repeat
                y = y.view(base_batch, n_repeat, *y.shape[1:])[:, 0]
                x_grouped = x.view(base_batch, n_repeat, *x.shape[1:])

                logits_sum = None
                for repeat_idx in range(n_repeat):
                    current_logits = model(x_grouped[:, repeat_idx])
                    if logits_sum is None:
                        logits_sum = {task: value.clone() for task, value in current_logits.items()}
                    else:
                        for task, value in current_logits.items():
                            logits_sum[task] += value
                logits = {task: value / n_repeat for task, value in logits_sum.items()}
            else:
                logits = model(x)

            task_masks = build_multilabel_task_masks(
                y,
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
                y_task = y[:, task_idx, :n_steps]
                logits_task = logits[task]
                sample_mask = task_masks[task]

                if logits_task.shape[1] != n_steps or logits_task.shape[2] != n_classes:
                    raise ValueError(
                        f"Task '{task}' logits shape {tuple(logits_task.shape)} does not match expected "
                        f"(B, {n_steps}, {n_classes})."
                    )
                if sample_mask.ndim != 1 or sample_mask.shape[0] != y_task.shape[0]:
                    raise ValueError(
                        f"Task '{task}' mask shape {tuple(sample_mask.shape)} does not match expected (B,)."
                    )
                if sample_mask.sum() == 0:
                    continue

                logits_selected = logits_task[sample_mask]
                y_selected = y_task[sample_mask]
                if (y_selected < 0).any():
                    raise ValueError(f"Task '{task}' contains invalid targets. Unclear labels must be filtered in get_target().")
                if (y_selected >= n_classes).any():
                    raise ValueError(f"Task '{task}' contains out-of-range targets.")

                loss_task = self.task_loss_functions[task](logits_selected.transpose(1, 2), y_selected)
                loss_task = self.task_loss_weights[task] * loss_task
                pred_selected = logits_selected.argmax(dim=-1)
                pred_flat = pred_selected.reshape(-1)
                y_flat = y_selected.reshape(-1)

                per_task_losses[task_idx] += float(loss_task.item())
                per_task_loss_batches[task_idx] += 1
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

            cnt += 1
            loss_sum += float(loss.item())

            if self.log_batches and batch_cms:
                self._log_from_cms(batch_cms, float(loss.item()), mode=mode, scope="batch", step=self.steps[mode])

            task_desc_parts = []
            for task_idx, task in enumerate(self.task_specs):
                if per_task_loss_batches[task_idx] == 0 or per_task_counts[task_idx] == 0:
                    task_desc_parts.append(f"{task['task'][:5]}: n/a")
                    continue
                task_loss = per_task_losses[task_idx] / per_task_loss_batches[task_idx]
                task_acc = per_task_correct[task_idx] / per_task_counts[task_idx] * 100.0
                task_desc_parts.append(f"{task['task'][:5]}: {task_loss:2.4f} - {task_acc:2.2f}")
            task_desc = " ".join(task_desc_parts)
            desc = f"{prefix:<12} loss {loss_sum/cnt:2.4f} {task_desc}"

            self.steps[mode] += 1
            logger.progress_status(desc)
            logger.progress_advance(loader.batch_size)

        logger.progress_close()
        epoch_loss = loss_sum / max(cnt, 1)
        epoch_cms_np = {
            task: cm.detach().cpu().numpy()
            for task, cm in epoch_cms.items()
        }
        self._log_from_cms(epoch_cms_np, epoch_loss, mode=mode, scope="epoch", step=self.epoch_step)
        self.log_confusion_tables(mode, epoch_cms_np)

        return epoch_loss, epoch_cms_np
