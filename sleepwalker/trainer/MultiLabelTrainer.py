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
from sklearn.metrics import confusion_matrix
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.trainer.utils import (
    cohen_kappa_from_confusion_matrix,
    f1_score_from_confusion_matrix,
    store_checkpoint,
)
from sleepwalker.utils import logger


class MultiLabelTrainer(ABC):
    """
    Multi-task classification with one softmax head per task.

    task_config structure:
        {
            "sleep staging": {"labels": ["wake", "n1", "n2", "n3", "rem"], "default": None, "percentage": 0.5},
            "breathing": {"labels": ["apnea", "hypopnea", "regular breathing"], "default": "regular breathing", "percentage": 0.5},
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
        early_stopping: Optional[int] = None
    ):
        self.epochs = epochs
        self.save_every = save_every
        self.early_stopping_patience = early_stopping
        self.device = device
        self.warmup_device = warmup_device

        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler
        self.loss_function = loss_function

        self.task_config = OrderedDict()
        self.task_slices = {}
        self.task_defaults = {}
        self.classes = []

        used_classes = set()
        offset = 0
        for task, cfg in task_config.items():
            labels = list(cfg["labels"])
            if "default" not in cfg:
                raise ValueError(f"Task '{task}' is missing a default entry.")
            default = cfg["default"]
            percentage = cfg.get("percentage", 0.5)

            if len(labels) == 0:
                raise ValueError(f"Task '{task}' does not contain any labels.")

            for label in labels:
                if label in used_classes:
                    raise ValueError(f"Class '{label}' occurs in multiple tasks.")
                used_classes.add(label)

            if default is not None and default not in labels:
                raise ValueError(f"Default label '{default}' is not part of task '{task}'.")
            if percentage <= 0 or percentage > 1:
                raise ValueError(f"Task '{task}' has invalid percentage '{percentage}'.")

            self.task_config[task] = {"labels": labels, "default": default, "percentage": percentage}
            self.task_slices[task] = slice(offset, offset + len(labels))
            self.classes.extend(labels)
            offset += len(labels)

        self.num_classes = len(self.classes)

    @staticmethod
    def target_to_multilabel(targets: dict[str, float], task_config: dict[str, dict], total_event_seconds, raise_error=True):
        out = []

        for task, cfg in task_config.items():
            labels = cfg["labels"]
            default = cfg["default"]
            min_event_seconds = total_event_seconds * cfg.get("percentage", 0.5)

            active = torch.tensor(
                [float(targets.get(label, 0.0)) > min_event_seconds for label in labels],
                dtype=torch.bool,
            )
            active_sum = int(active.sum().item())
            task_target = torch.zeros(len(labels), dtype=torch.float)

            if raise_error and active_sum > 1:
                raise ValueError(f"Multiple active classes found for task '{task}'.")

            if active_sum == 1:
                task_target[int(active.nonzero(as_tuple=False).item())] = 1.0
            elif active_sum == 0 and default is not None:
                task_target[labels.index(default)] = 1.0
            elif active_sum == 0 and raise_error:
                raise ValueError(f"Ambiguous class labels found for task '{task}'.")

            out.append(task_target)

        return torch.cat(out, dim=0)

    @staticmethod
    def get_item(patient, time, data, target, target_extra=None, class_cnts: Optional[List[float]] = None, task_config: Optional[dict[str, dict]] = None):
        if task_config is None:
            raise ValueError("task_config must not be None.")

        try:
            freq = pd.to_timedelta(target.index.freq).total_seconds()
            targets = {str(k): float(v) for k, v in target.sum().to_dict().items()}
            target = MultiLabelTrainer.target_to_multilabel(targets, task_config, len(target)*freq, True)

            item = {"patient": patient, "time": time, "target": target}

            if class_cnts and len(class_cnts) == len(target):
                probas = class_cnts / np.sum(class_cnts)
                m = min(probas)
                idx = target.nonzero(as_tuple=False).flatten().tolist()
                if len(idx) > 0 and random.random() > min([m / probas[i] for i in idx]):
                    return None

            if target_extra is not None:
                freq = pd.to_timedelta(target_extra.index.freq).total_seconds()
                targets = {str(k): float(v) for k, v in target_extra.sum().to_dict().items()}
                item["target_extra"] = MultiLabelTrainer.target_to_multilabel(targets, task_config, len(target_extra)*freq, False)

            item["data"] = torch.from_numpy(data.values).float()
            return item
        except Exception:
            pass
        return None

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
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)

        loss_sum = 0
        cm_sum = {
            task: np.zeros((len(cfg["labels"]), len(cfg["labels"])), dtype=np.int64)
            for task, cfg in self.task_config.items()
        }
        cnt = 0
        n_samples = 0
        t0 = time.perf_counter()

        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"

        for batch in loader:
            x = batch["data"].to(self.device)
            y = batch["target"].to(self.device)

            if opt is not None:
                opt.zero_grad(set_to_none=True)

            logits = model(x)

            losses = []
            cms = {}
            for task, task_slice in self.task_slices.items():
                y_task = y[:, task_slice]
                logits_task = logits[:, task_slice]

                losses.append(self.loss_function(logits_task, y_task))

                target_np = y_task.argmax(axis=1).cpu().numpy()
                pred_np = logits_task.argmax(axis=1).cpu().numpy()
                cms[task] = confusion_matrix(target_np, pred_np, labels=range(logits_task.shape[1]))

            loss = sum(losses)

            if opt is not None:
                loss.backward()
                opt.step()

            for task, cm in cms.items():
                cm_sum[task] += cm

            cnt += 1
            n_samples += y.shape[0]
            loss_sum += float(loss.item()) / len(self.task_config)

            step = self.steps[mode]
            self._log_from_cms(cms, float(loss.item()) / len(self.task_config), mode=mode, scope="batch", step=step)

            accs = np.mean([
                cm.trace() / cm.sum() * 100.0
                for cm in cm_sum.values()
                if cm.sum() > 0
            ])
            f1_micro = np.mean([
                f1_score_from_confusion_matrix(cm, macro=False)
                for cm in cm_sum.values()
                if cm.sum() > 0
            ])
            f1_macro = np.mean([
                f1_score_from_confusion_matrix(cm, macro=True)
                for cm in cm_sum.values()
                if cm.sum() > 0
            ])
            coehns_kappa = np.mean([
                cohen_kappa_from_confusion_matrix(cm)
                for cm in cm_sum.values()
                if cm.sum() > 0
            ])

            desc = f"{prefix:<12} {loss_sum/cnt:2.4f} acc {accs:2.3f} " \
                   f"f1 (mi/ma) {f1_micro:1.4f}/{f1_macro:1.4f} κ {coehns_kappa:2.3f}"

            self.steps[mode] += 1
            logger.progress_status(desc)
            logger.progress_advance(loader.batch_size)

        logger.progress_close()
        epoch_loss = loss_sum / max(cnt, 1)
        self._log_from_cms(cm_sum, epoch_loss, n_samples / max(time.perf_counter() - t0, 1e-8), mode=mode, scope="epoch", step=self.epoch_step)

        for task, cm in cm_sum.items():
            logger.info(f"{mode.upper()} | task={task} | labels={self.task_config[task]['labels']} | confusion_matrix=\n{cm}")

        return epoch_loss, cm_sum

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
