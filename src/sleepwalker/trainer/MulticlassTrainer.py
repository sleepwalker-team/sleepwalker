"""Concrete trainer for single-head multiclass tasks.

This trainer is used by several task-specific scripts in the repository,
including sleep staging and event-detection variants that emit one or more
categorical targets per window. Tests cover class balancing behavior,
repeated-window evaluation, and prediction-frame generation.
"""

from functools import partial
import random
from typing import Callable, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import torch

from sleepwalker.trainer.BaseTrainer import BaseTrainer
from sleepwalker.trainer.losses import class_weights_for_loss
from sleepwalker.utils import logger
from sleepwalker.datasets.utils import estimate_class_cnts
from sleepwalker.trainer.utils.display import format_confusion_table, render_confusion_table_grid
from sleepwalker.metrics import accuracy_from_confusion_matrix, cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix


class MulticlassTrainer(BaseTrainer):
    """Train and evaluate a single softmax classification head.

    Args:
        epochs: Number of training epochs.
        optimizer: Factory that builds an optimizer for the model.
        classes: Ordered output labels used for targets, confusion matrices,
            and prediction tables.
        loss_function: Base loss function, usually cross entropy.
        target_resolution: Duration covered by all predictions from one input
            window. Target selection itself remains in ``prepare_target``.
        device: Torch device used for training and inference.
        warmup_device: Device used during preprocessor warmup.
        save_every: Checkpoint cadence in epochs.
        eval_every: Validation cadence in epochs.
        lr_scheduler: Optional scheduler factory.
        early_stopping: Optional validation patience in epochs.
        return_best: Whether to return the lowest-loss validation checkpoint
            instead of leaving the model at its last training epoch.
        train_transform: Optional list of per-sample transforms.
        loss_mode: Loss reweighting mode understood by
            ``class_weights_for_loss``.
        class_weights: Optional manual per-class weights.
        class_counts: Optional fixed raw counts used instead of traversing the
            training loader during warmup.
        balance_batches: Whether to reject overrepresented targets in
            ``prepare_target`` during training.
        balance_gamma: Exponent applied to the inverse-frequency acceptance
            probability during batch balancing. ``1.0`` reproduces the
            previous behavior, values above ``1.0`` make balancing more
            aggressive, and values between ``0`` and ``1`` make it weaker.
        sequence_len: Number of categorical predictions emitted per window.
        target_offset: Timestamp of the first prediction relative to the input
            window start. Target selection itself remains in ``prepare_target``.
    """

    def __init__(
        self,
        epochs: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        classes: list[str],
        loss_function: Callable,
        target_resolution: str | pd.Timedelta,
        device: str = "cuda:0",
        warmup_device: str = "cpu",
        save_every: int = 10,
        eval_every: int = 10,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
        early_stopping: Optional[int] = None,
        return_best: bool = True,
        train_transform: Optional[list[Callable]] = None,
        loss_mode: str = "regular",
        class_weights: Optional[dict[str, float]] = None,
        class_counts: Optional[dict[str, float]] = None,
        balance_batches: bool = False,
        balance_gamma: float = 1.0,
        sequence_len: int = 1,
        target_offset: str | pd.Timedelta = "0s",
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
            train_transform=train_transform,
        )

        self.classes = classes
        self.num_classes = len(classes)
        self.loss_function = loss_function
        self.base_loss_function = loss_function
        self.loss_mode = loss_mode
        self.class_weights = dict(class_weights or {})
        self.class_counts = None if class_counts is None else {label: float(count) for label, count in class_counts.items()}
        if self.class_counts is not None:
            if set(self.class_counts) != set(self.classes):
                raise ValueError(f"class_counts must contain exactly {self.classes}, got {sorted(self.class_counts)}.")
            if any(count <= 0 for count in self.class_counts.values()):
                raise ValueError("class_counts values must be positive.")
        self.balance_batches = balance_batches
        self.balance_gamma = float(balance_gamma)
        self.sequence_len = int(sequence_len)
        if self.sequence_len < 1:
            raise ValueError("sequence_len must be at least 1.")
        self.target_resolution = pd.to_timedelta(target_resolution)
        if self.target_resolution <= pd.Timedelta(0):
            raise ValueError("target_resolution must be positive.")
        self.target_offset = pd.to_timedelta(target_offset)
        if self.target_offset < pd.Timedelta(0):
            raise ValueError("target_offset must not be negative.")

    def classification_contract(self) -> dict:
        return {
            "type": "single-head-multiclass",
            "classes": list(self.classes),
            "sequence_len": self.sequence_len,
            "target_resolution": str(self.target_resolution),
            "target_offset": str(self.target_offset),
        }

    def _keep_balanced_target(self, target, class_cnts: list[float]) -> bool:
        target_arr = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)
        if target_arr.ndim == 0:
            target_indices = [int(target_arr.item())]
        elif target_arr.ndim == 1:
            target_indices = [int(np.argmax(target_arr))]
        elif target_arr.ndim == 2:
            target_indices = np.argmax(target_arr, axis=-1).tolist()
        else:
            raise ValueError(f"Expected multiclass target with ndim <= 2, got shape {target_arr.shape}.")

        probas = np.asarray(class_cnts, dtype=float)
        if probas.ndim != 1 or len(probas) == 0:
            raise ValueError("class_cnts must be a non-empty 1D sequence.")
        if any(target_idx < 0 or target_idx >= len(probas) for target_idx in target_indices):
            raise ValueError(f"Target indices {target_indices} out of range for {len(probas)} classes.")

        probas = probas / probas.sum()
        raw_keep_prob = min(float(np.clip(probas.min() / probas[target_idx], 0.0, 1.0)) for target_idx in target_indices)
        keep_prob = float(np.clip(raw_keep_prob ** self.balance_gamma, 0.0, 1.0))
        return random.random() <= keep_prob

    def _wrap_prepare_target_for_balancing(self, prepare_target, class_cnts: list[float]):
        def wrapped_prepare_target(*args, **kwargs):
            prepared_target = prepare_target(*args, **kwargs)
            if prepared_target is None:
                return None
            if not isinstance(prepared_target, dict):
                raise ValueError(f"prepare_target must return dict or None, but received {type(prepared_target)}.")
            if "target" not in prepared_target:
                raise ValueError("Balanced multiclass training requires prepare_target to return a 'target' entry.")
            if not self._keep_balanced_target(prepared_target["target"], class_cnts):
                return None
            return prepared_target

        return wrapped_prepare_target

    def warmup_trainer(self, data_loader, device: str = "cuda"):
        """Prepare class weighting and optional batch balancing.

        Notes:
            Batch balancing only applies to datasets that expose a mutable
            ``prepare_target_callback``. Tests explicitly cover the repeat and
            balancing paths.
        """
        dataset = data_loader.dataset

        class_cnts = self.class_counts
        if (self.balance_batches or self.loss_mode != "regular") and class_cnts is None:
            class_cnts = estimate_class_cnts(data_loader)
            logger.info(f"Class counts are {class_cnts}")
        elif class_cnts is not None:
            logger.info(f"Using configured class counts: {class_cnts}")
            
        if self.balance_batches and class_cnts is not None:
            current_datasets = dataset.datasets if hasattr(dataset, "datasets") else [dataset]
            can_balance_batches = True
            for current_ds in current_datasets:
                if not hasattr(current_ds, "prepare_target_callback"):
                    logger.warning(
                        "Ignoring balance_batches because the training dataset does not expose "
                        f"a mutable prepare_target_callback. Found {type(current_ds).__name__}. "
                        "Batch balancing only works for live EDF-backed datasets, not frozen numpy caches."
                    )
                    can_balance_batches = False
                    break
                base_callback = getattr(current_ds, "_base_prepare_target_callback", current_ds.prepare_target_callback)
                if base_callback is None:
                    raise ValueError("balance_batches requires a prepare_target callback on the training dataset.")
            if can_balance_batches:
                # TODO: Rebuild the loader (or implement balancing in a sampler)
                # before enabling this path. Persistent DataLoader workers hold
                # their own dataset copies, so mutating the callback here after
                # class-count estimation does not affect those workers.
                for current_ds in current_datasets:
                    base_callback = getattr(current_ds, "_base_prepare_target_callback", current_ds.prepare_target_callback)
                    current_ds._base_prepare_target_callback = base_callback
                    current_ds.prepare_target_callback = self._wrap_prepare_target_for_balancing(
                        base_callback,
                        [class_cnts.get(c, 1.0) for c in current_ds.get_classes()],
                    )

        if self.loss_mode != "regular":
            if class_cnts is None:
                raise ValueError("class counts are required for non-regular multiclass loss modes.")
            class_weights = class_weights_for_loss(self.class_weights, class_cnts, self.loss_mode)
        else:
            class_weights = self.class_weights

        if len(class_weights) > 0:
            weights_torch = torch.tensor([class_weights.get(c, 1) for c in dataset.get_classes()], device=self.device)
            self.loss_function = partial(self.base_loss_function, weight=weights_torch)
        else:
            self.loss_function = self.base_loss_function

    def _log_from_cm(self, cm: np.ndarray, loss_value: float, mode: str, scope: str = "batch", step:int = 0, show_cm:bool = False):
        """Centralized metric logging from confusion matrix."""  
        total = cm.sum()  
        if total == 0:  
            return  
        
        acc = accuracy_from_confusion_matrix(cm)
        f1_micro = f1_score_from_confusion_matrix(cm, macro=False)  
        f1_macro = f1_score_from_confusion_matrix(cm, macro=True)  
        kappa = cohen_kappa_from_confusion_matrix(cm)  
        logger.metric(f"{scope}/{mode}/accuracy", acc, step=step)  
        logger.metric(f"{scope}/{mode}/f1_micro", f1_micro, step=step)  
        logger.metric(f"{scope}/{mode}/f1_macro", f1_macro, step=step)  
        logger.metric(f"{scope}/{mode}/coehns_kappa", kappa, step=step)  
        logger.metric(f"{scope}/{mode}/loss", float(loss_value), step=step) 
        if show_cm:
            fig, ax = plt.subplots(figsize=(max(6, len(self.classes)), max(5, len(self.classes) * 0.8)))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
            disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
            ax.set_title(f"{scope} confusion matrix {mode} {step} ")
            fig.tight_layout()
            logger.figure(f"{mode}_cm_{scope}_{step}", fig)
            plt.close(fig)
            logger.info(render_confusion_table_grid([format_confusion_table(self.classes, cm)], header=f"{mode.upper()} confusion matrix", n_cols=1))

    def run_epoch(self, loader, opt, model, prefix="", lr_scheduler=None):
        """Run one train, validation, or test epoch.

        Args:
            loader: Dataloader producing batch dictionaries with ``data`` and
                one-hot ``target`` fields.
            opt: Optimizer for training epochs, or ``None`` during evaluation.
            model: Model instance to execute.
            prefix: Progress-label prefix used to infer the logging mode.

        Returns:
            A tuple ``(epoch_loss, confusion_matrix)``.
        """
        target_step_resolution = self.target_resolution / self.sequence_len
        if hasattr(model, "epoch_len_s") and model.epoch_len_s is not None:
            model_step_resolution = pd.to_timedelta(model.epoch_len_s, unit="s")
            if model_step_resolution != target_step_resolution:
                raise ValueError(f"Model epoch_len={model_step_resolution} does not match target_resolution / sequence_len={target_step_resolution}.")
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)
        nc = self.num_classes
        
        loss_sum = 0
        loss_weight_sum = 0
        cm_sum = np.zeros((nc, nc), dtype=np.int64)
        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"
        for batch in loader:
            if batch is None:
                continue
            if opt is not None:
                opt.zero_grad(set_to_none=True)

            x = batch["data"].to(self.device, non_blocking=True)
            if opt is not None:
                x = self.apply_train_transform(x)
            logits = model(x)
            y = batch["target"].to(self.device)

            expected_shape = (y.shape[0], self.sequence_len, self.num_classes)
            if tuple(logits.shape) != expected_shape:
                raise ValueError(f"Expected model logits shaped {expected_shape} ([B, S, C]), got {tuple(logits.shape)}.")
            if tuple(y.shape) != expected_shape:
                raise ValueError(f"Expected targets shaped {expected_shape} ([B, S, C]), got {tuple(y.shape)}. Configure the dataset prepare_target callback with the same sequence_len as the trainer.")
            target_mask = batch.get("target_mask")
            if target_mask is None:
                loss = self.loss_function(logits.transpose(1, 2), y.transpose(1, 2))
                target_mask = torch.ones(y.shape[:2], dtype=torch.bool, device=self.device)
            else:
                target_mask = target_mask.to(self.device, dtype=torch.bool)
                if tuple(target_mask.shape) != tuple(y.shape[:2]):
                    raise ValueError(f"Expected target_mask shaped {tuple(y.shape[:2])} ([B, S]), got {tuple(target_mask.shape)}.")
                step_loss = self.loss_function(logits.transpose(1, 2), y.transpose(1, 2), reduction="none")
                if tuple(step_loss.shape) != tuple(target_mask.shape):
                    raise ValueError(f"Masked multiclass loss must return [B, S], got {tuple(step_loss.shape)}.")
                loss = step_loss[target_mask].mean()

            pred_np = logits.argmax(dim=-1)[target_mask].cpu().numpy()
            target_np = y.argmax(dim=-1)[target_mask].cpu().numpy()
            
            if opt is not None:
                loss.backward()
                opt.step()

            cm = confusion_matrix(target_np, pred_np, labels=range(len(self.classes)))
            cm_sum += cm
            loss_weight = int(target_mask.sum().item())
            loss_sum += float(loss.item()) * loss_weight
            loss_weight_sum += loss_weight

            step = self.steps[mode]
            self._log_from_cm(cm, float(loss.item()), mode=mode, scope="batch", step=step, show_cm = False) 

            accs = cm_sum.trace() / cm_sum.sum() * 100.0
            f1_micro = f1_score_from_confusion_matrix(cm_sum, macro=False)
            f1_macro = f1_score_from_confusion_matrix(cm_sum, macro=True)
            coehns_kappa = cohen_kappa_from_confusion_matrix(cm_sum)

            desc = f"{prefix:<12} {loss_sum/loss_weight_sum:2.4f} acc {accs:2.3f} " \
                   f"f1 (mi/ma) {f1_micro:1.4f}/{f1_macro:1.4f} κ {coehns_kappa:2.3f}"
            
            self.steps[mode] += 1
            logger.progress_status(desc)
            logger.progress_advance(loader.batch_size)
            if lr_scheduler is not None:
                lr_scheduler.step()

        logger.progress_close()
        epoch_loss = loss_sum / max(loss_weight_sum, 1)
        self._log_from_cm(cm_sum, epoch_loss, mode=mode, scope="epoch", step=self.epoch_step, show_cm = True)  
        

        return epoch_loss, cm_sum  
