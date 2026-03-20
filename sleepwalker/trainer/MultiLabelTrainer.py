import os
import random
import shutil
import tempfile
import time
from abc import ABC
from collections import OrderedDict
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.models.MetaModel import MetaModel
from sleepwalker.trainer.utils import (
    cohen_kappa_from_confusion_matrix,
    f1_score_from_confusion_matrix,
    store_checkpoint,
)
from sleepwalker.utils import logger


class MultiLabelTrainer(ABC):
    """
    Multi-task classification with one softmax head per task and optional
    multiple resolution steps per task inside the dataset target window.

    task_config structure:
        {
            "sleep staging": {"labels": ["wake", "n1", "n2", "n3", "rem"], "default": None, "percentage": 0.5, "target_resolution": "30s"},
            "breathing": {"labels": ["apnea", "hypopnea", "regular breathing"], "default": "regular breathing", "percentage": 0.5, "target_resolution": "10s"},
        }

    `default` is mandatory and may be None.
    `percentage` is optional and defaults to 0.5.
    """

    def __init__(
        self,
        epochs: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        task_config: dict[str, dict],
        loss_function: Callable,
        device: str = "cuda:0",
        warmup_device: str = "cpu",
        save_every: int = 1,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
        early_stopping: Optional[int] = None,
        log_batches: bool = True
    ):
        self.epochs = epochs
        self.save_every = save_every
        self.early_stopping_patience = early_stopping
        self.device = device
        self.warmup_device = warmup_device
        self.log_batches = log_batches

        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler
        self.loss_function = loss_function

        self.task_config = OrderedDict(task_config)
        self.task_specs = list(self.task_config.values())
        self.largest_task_resolution = max(cfg["target_resolution"] for cfg in self.task_specs)
        self.max_task_steps = max(cfg["n_steps"] for cfg in self.task_specs)
        self.classes = [label for cfg in self.task_config.values() for label in cfg["labels"]]
        self.num_classes = len(self.classes)
        self.num_tasks = len(self.task_specs)

    @staticmethod
    def normalize_task_config(task_config: dict[str, dict]):
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
            }
            normalized[task] = spec

        return normalized

    @staticmethod
    def task_to_multiclass(targets: dict[str, float], cfg: dict, total_event_seconds, raise_error=True) -> int:
        labels = cfg["labels"]
        default = cfg["default"]
        min_event_seconds = total_event_seconds * cfg.get("percentage", 0.5)

        active = torch.tensor(
            [float(targets.get(label, 0.0)) > min_event_seconds for label in labels],
            dtype=torch.bool,
        )
        active_sum = int(active.sum().item())

        if raise_error and active_sum > 1:
            raise ValueError(f"Multiple active classes found for task.")

        if active_sum == 1:
            return int(active.nonzero(as_tuple=False).item())
        if active_sum == 0 and default is not None:
            return labels.index(default)
        if active_sum == 0 and raise_error:
            raise ValueError("Ambiguous class labels found for task.")
        return -1

    @staticmethod
    def target_to_multiclass(targets: pd.DataFrame, task_config: dict[str, dict], raise_error=True):
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
                    step_target = MultiLabelTrainer.task_to_multiclass(totals,cfg,step_seconds,raise_error)
                    if step_target < 0 and not raise_error:
                        return None
                    out[task_idx, step_idx] = step_target
                except ValueError as exc:
                    if raise_error:
                        raise ValueError(f"{exc} Task '{cfg['task']}', step {step_idx}.") from exc
                    return None

        return out

    @staticmethod
    def get_target(target, target_extra=None, class_cnts: Optional[List[float]] = None, task_config: Optional[dict[str, dict]] = None):
        if task_config is None:
            raise ValueError("task_config must not be None.")

        target = MultiLabelTrainer.target_to_multiclass(target, task_config, raise_error=False)
        if target is None:
            return None

        item = {"target": target}

        if class_cnts and len(class_cnts) == sum(len(cfg["labels"]) for cfg in task_config.values()):
            probas = class_cnts / np.sum(class_cnts)
            m = min(probas)
            idx = []
            offset = 0
            for task_idx, cfg in enumerate(task_config.values()):
                idx.extend([offset + int(i) for i in target[task_idx].tolist() if i >= 0])
                offset += len(cfg["labels"])
            if len(idx) > 0 and random.random() > min([m / probas[i] for i in idx]):
                return None

        if target_extra is not None:
            extra = MultiLabelTrainer.target_to_multiclass(target_extra, task_config, raise_error=False)
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

    def format_confusion_table(self, task: str, cm: np.ndarray) -> list[str]:
        labels = list(self.task_config[task]["labels"])
        short_labels = [label[:5] for label in labels]
        df = pd.DataFrame(cm, index=short_labels, columns=short_labels)
        table_lines = df.to_string().splitlines()

        total = cm.sum()
        if total > 0:
            accuracy = cm.trace() / total * 100.0
            f1_micro = f1_score_from_confusion_matrix(cm, macro=False)
            f1_macro = f1_score_from_confusion_matrix(cm, macro=True)
        else:
            accuracy = 0.0
            f1_micro = 0.0
            f1_macro = 0.0

        lines = [
            f"{task}",
            *table_lines,
            f"acc: {accuracy:5.2f}% f1(mi/ma): {f1_micro:1.4f} / {f1_macro:1.4f} ",
        ]

        width = max(len(line) for line in lines)
        return [line.ljust(width) for line in lines]

    def log_confusion_tables(self, mode: str, cm_sum: dict[str, np.ndarray]):
        blocks = [self.format_confusion_table(task, cm) for task, cm in cm_sum.items()]
        if len(blocks) == 0:
            return

        separator = "   "
        output_lines = [f"{mode.upper()} confusion matrices"]
        for start in range(0, len(blocks), 3):
            row_blocks = blocks[start:start + 3]
            height = max(len(block) for block in row_blocks)
            padded_blocks = [block + [" " * len(block[0])] * (height - len(block)) for block in row_blocks]
            output_lines.extend([
                separator.join(block[line_idx] for block in padded_blocks).rstrip()
                for line_idx in range(height)
            ])
            if start + 3 < len(blocks):
                output_lines.append("")
        logger.info("\n".join(output_lines))

    def warmup_preprocessors(self, model: BaseModel, data_loader:DataLoader, device:str = "cuda") -> BaseModel:
        if hasattr(model, "_warmup_preprocessors"):
            return model._warmup_preprocessors(data_loader, device)

        model.to(device)
        total_batches = len(data_loader)
        batch_size = data_loader.batch_size

        if batch_size is None:
            raise ValueError(f"batch_size should not be None here.")

        for idx in range(len(model.preprocessors)):
            logger.progress_start(total_batches*batch_size, desc=f" {idx}/{len(model.preprocessors) - 1}", leave=True)
            if model.preprocessors[idx].requires_warmup():
                for batch in data_loader:
                    x = batch["data"].to(device)
                    x = model.apply_preprocessors(x, idx)
                    model.preprocessors[idx].update(x)
                    logger.progress_advance(batch_size)
            else:
                logger.progress_advance(total_batches*batch_size)
            logger.progress_close()

        return model

    def run_epoch(self, loader, opt, model, prefix=""):
        if not isinstance(model, MetaModel):
            raise ValueError("MultiLabelTrainer requires a MetaModel.")
        dataset_resolution = getattr(loader.dataset, "target_resolution", None)
        if dataset_resolution is None:
            raise ValueError("MultiLabelTrainer requires loader.dataset.target_resolution.")
        dataset_resolution = pd.to_timedelta(dataset_resolution)
        if dataset_resolution != self.largest_task_resolution:
            raise ValueError(
                f"Dataset target_resolution={dataset_resolution} does not match largest_task_resolution={self.largest_task_resolution}."
            )
        
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)

        loss_sum = 0
        epoch_cms = {
            cfg["task"]: torch.zeros((len(cfg["labels"]), len(cfg["labels"])),dtype=torch.int64,device=self.device,) for cfg in self.task_specs
        }
        cnt = 0
        n_samples = 0
        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"

        per_task_losses = [0 for _ in range(len(self.task_specs))]
        per_task_acc = [0 for _ in range(len(self.task_specs))]
        for batch in loader:
            x = batch["data"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)

            if opt is not None:
                opt.zero_grad(set_to_none=True)

            logits = model(x)

            losses = []
            batch_cms = {} if self.log_batches else None
            for task_idx, cfg in enumerate(self.task_specs):
                task = cfg["task"]
                n_classes = len(cfg["labels"])
                n_steps = cfg["n_steps"]
                y_task = y[:, task_idx, :n_steps]
                logits_task = logits[task]

                if logits_task.shape[1] != n_steps or logits_task.shape[2] != n_classes:
                    raise ValueError(
                        f"Task '{task}' logits shape {tuple(logits_task.shape)} does not match expected "
                        f"(B, {n_steps}, {n_classes})."
                    )

                logits_flat = logits_task.reshape(-1, logits_task.shape[-1])
                y_flat = y_task.reshape(-1)
                if (y_flat < 0).any():
                    raise ValueError(f"Task '{task}' contains invalid targets. Unclear labels must be filtered in get_target().")
                if (y_flat >= n_classes).any():
                    raise ValueError(f"Task '{task}' contains out-of-range targets.")

                
                loss_task = self.loss_function(logits_flat, y_flat)
                per_task_losses[task_idx] += loss_task.item()

                pred_flat = logits_task.argmax(dim=-1).reshape(-1)
                per_task_acc[task_idx] += (pred_flat == y_flat).float().mean().item() * 100.0

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
            n_samples += y.shape[0]
            loss_sum += float(loss.item())

            if self.log_batches and batch_cms:
                self._log_from_cms(batch_cms, float(loss.item()), mode=mode, scope="batch", step=self.steps[mode])

            task_desc = " ".join(
                f"{task['task'][:5]}: {task_loss/cnt:2.4f} - {task_acc/cnt:2.2f}"
                for task, task_loss, task_acc in zip(self.task_specs, per_task_losses, per_task_acc)
            )
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

    def test(self, model: BaseModel, test_loader):
        model.eval()
        with torch.inference_mode():
            test_loss, test_cm = self.run_epoch(test_loader, None, model, f"TEST")
        return test_loss, test_cm

    def fit(self, model: BaseModel, train_loader, val_loader = None):
        opt = self.optimizer_fn(model)

        if self.lr_scheduler_fn is not None:
            lr_scheduler = self.lr_scheduler_fn(opt)
            if isinstance(lr_scheduler, OneCycleLR):
                raise ValueError(f"OneCycleLR is currently not supported")
        else:
            lr_scheduler = None

        if self.early_stopping_patience and val_loader is None:
            logger.warning(f"early_stopping was set to true, but no validation dataset was given. Disabling early stopping")
            self.early_stopping_patience = None

        logger.context("Warmup preprocessors")
        self.warmup_preprocessors(model, train_loader, self.warmup_device)
        logger.uncontext()

        model = model.to(self.device)
        val_losses: list[float] = []
        losses = []
        cms = []

        self.best_model_idx = None
        self.best_checkpoint = None
        self.steps = {"train":0, "val":0, "test":0}
        self.epoch_step = 0

        for epoch in range(self.epochs):
            model.train()
            loss, cm = self.run_epoch(train_loader, opt, model, f"TRAIN [{epoch+1}/{self.epochs}]")
            cms.append({"train":cm})
            losses.append({"train":loss})

            if self.save_every > 0 and (epoch % self.save_every == 0):
                logger.info(f"Logging intermediate model after {epoch} epochs.")

                folder = store_checkpoint(model, opt, lr_scheduler, tempfile.mkdtemp(prefix=f"checkpoint_epoch_{epoch}_"))
                logger.artifact(path=os.path.join(folder, "model.pt"), dest=f"{epoch}")
                logger.artifact(path=os.path.join(folder, "optimizer.pt"), dest=f"{epoch}")
                if lr_scheduler:
                    logger.artifact(path=os.path.join(folder, "scheduler.pt"), dest=f"{epoch}")

            if lr_scheduler is not None:
                lr_scheduler.step()

            if val_loader is not None:
                model.eval()
                with torch.inference_mode():
                    val_loss, val_cm = self.run_epoch(val_loader, None, model, f"VAL [{epoch+1}/{self.epochs}]")
                cms[-1]["val"] =  val_cm
                losses[-1]["val"] =  val_loss
                val_losses.append(val_loss)

                imin = np.argmin(val_losses)
                if self.best_model_idx is None or imin != self.best_model_idx:
                    if self.best_checkpoint is not None:
                        logger.info(f"Found old best model in {self.best_checkpoint}. Deleting it")
                        shutil.rmtree(self.best_checkpoint)

                    self.best_checkpoint = store_checkpoint(model, opt, lr_scheduler, tempfile.mkdtemp(prefix="sleepwalker_best_model_"))
                    self.best_model_idx = imin

                if self.early_stopping_patience and (epoch - imin >= self.early_stopping_patience):
                    logger.info(f"Early stopping after {epoch} epochs - best epoch was {imin}")
                    return {
                        "losses":losses,
                        "cms":cms,
                        "best_model":imin,
                        "checkpoint":self.best_checkpoint
                    }

            self.epoch_step += 1

        return { "losses":losses, "cms":cms }
