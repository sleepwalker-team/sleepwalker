"""Concrete trainer for single-head multiclass tasks.

This trainer is used by several task-specific scripts in the repository,
including sleep staging and event-detection variants that reduce each window to
one categorical target. Tests cover class balancing behavior, repeated-window
evaluation, and prediction-frame generation.
"""

from functools import partial
import random
from typing import Callable, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch

from sleepwalker.trainer.BaseTrainer import BaseTrainer
from sleepwalker.trainer.losses import class_weights_for_loss
from sleepwalker.utils import logger
from sleepwalker.datasets.utils import RepeatSampler
from sleepwalker.datasets.utils import estimate_class_cnts
from sleepwalker.trainer.utils.display import format_confusion_table, render_confusion_table_grid
from sleepwalker.trainer.utils.metrics import cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix


class MulticlassTrainer(BaseTrainer):
    """Train and evaluate a single softmax classification head.

    Args:
        epochs: Number of training epochs.
        optimizer: Factory that builds an optimizer for the model.
        classes: Ordered output labels used for targets, confusion matrices,
            and prediction tables.
        loss_function: Base loss function, usually cross entropy.
        device: Torch device used for training and inference.
        warmup_device: Device used during preprocessor warmup.
        save_every: Checkpoint cadence in epochs.
        lr_scheduler: Optional scheduler factory.
        early_stopping: Optional validation patience in epochs.
        train_transform: Optional list of per-sample transforms.
        n_repeat_train: Number of repeated views per training sample.
        n_repeat_test: Number of repeated views per evaluation sample.
        loss_mode: Loss reweighting mode understood by
            ``class_weights_for_loss``.
        class_weights: Optional manual per-class weights.
        balance_batches: Whether to reject overrepresented targets in
            ``prepare_target`` during training.
    """

    def __init__(
        self,
        epochs: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        classes: list[str],
        loss_function: Callable,
        device: str = "cuda:0",
        warmup_device: str = "cpu",
        save_every: int = 1,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
        early_stopping: Optional[int] = None,
        train_transform: Optional[list[Callable]] = None,
        n_repeat_train: int = 1,
        n_repeat_test: int = 1,
        loss_mode: str = "regular",
        class_weights: Optional[dict[str, float]] = None,
        balance_batches: bool = False,
    ):
        super().__init__(
            epochs=epochs,
            optimizer=optimizer,
            device=device,
            warmup_device=warmup_device,
            save_every=save_every,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            train_transform=train_transform,
            n_repeat_train=n_repeat_train,
            n_repeat_test=n_repeat_test,
        )

        self.classes = classes
        self.num_classes = len(classes)
        self.loss_function = loss_function
        self.base_loss_function = loss_function
        self.loss_mode = loss_mode
        self.class_weights = dict(class_weights or {})
        self.balance_batches = balance_batches

    def _keep_balanced_target(self, target, class_cnts: list[float]) -> bool:
        target_arr = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)
        if target_arr.ndim == 0:
            target_idx = int(target_arr.item())
        elif target_arr.ndim == 1:
            target_idx = int(np.argmax(target_arr))
        else:
            raise ValueError(f"Expected multiclass target with ndim <= 1, got shape {target_arr.shape}.")

        probas = np.asarray(class_cnts, dtype=float)
        if probas.ndim != 1 or len(probas) == 0:
            raise ValueError("class_cnts must be a non-empty 1D sequence.")
        if target_idx < 0 or target_idx >= len(probas):
            raise ValueError(f"Target index {target_idx} out of range for {len(probas)} classes.")

        probas = probas / probas.sum()
        keep_prob = float(np.clip(probas.min() / probas[target_idx], 0.0, 1.0))
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

        class_cnts = None
        if self.balance_batches or self.loss_mode != "regular":
            class_cnts = estimate_class_cnts(data_loader)
            logger.info(f"Class counts are {class_cnts}")
            
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

    def _prediction_frame(self, batch, outputs) -> pd.DataFrame:
        """Convert model logits into the standard multiclass prediction table."""
        probabilities = torch.softmax(outputs.detach().cpu(), dim=1)
        pred_idx = probabilities.argmax(dim=1)
        frame = {
            "patient": list(batch.get("patient", [None] * probabilities.shape[0])),
            "time": list(batch.get("time", [None] * probabilities.shape[0])),
            "prediction_idx": pred_idx.tolist(),
            "prediction": [self.classes[idx] for idx in pred_idx.tolist()],
        }
        for class_idx, label in enumerate(self.classes):
            frame[f"prob__{label}"] = probabilities[:, class_idx].tolist()
        return pd.DataFrame(frame)

    def _log_from_cm(self, cm: np.ndarray, loss_value: float, mode: str, scope: str = "batch", step:int = 0):
        """Centralized metric logging from confusion matrix."""  
        total = cm.sum()  
        if total == 0:  
            return  
        
        acc = cm.trace() / total * 100.0  
        f1_micro = f1_score_from_confusion_matrix(cm, macro=False)  
        f1_macro = f1_score_from_confusion_matrix(cm, macro=True)  
        kappa = cohen_kappa_from_confusion_matrix(cm)  
        logger.metric(f"{scope}/{mode}/accuracy", acc, step=step)  
        logger.metric(f"{scope}/{mode}/f1_micro", f1_micro, step=step)  
        logger.metric(f"{scope}/{mode}/f1_macro", f1_macro, step=step)  
        logger.metric(f"{scope}/{mode}/coehns_kappa", kappa, step=step)  
        logger.metric(f"{scope}/{mode}/loss", float(loss_value), step=step) 

    def run_epoch(self, loader, opt, model, prefix=""):
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
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)
        nc = self.num_classes
        
        loss_sum = 0
        cm_sum = np.zeros((nc, nc), dtype=np.int64)
        cnt = 0

        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"
        n_repeat = loader.sampler.n_repeat if isinstance(getattr(loader, "sampler", None), RepeatSampler) else 1

        for batch in loader:
            x = batch["data"].to(self.device)
            y = batch["target"].to(self.device)

            if opt is not None:
                x = self.apply_train_transform(x)
                opt.zero_grad(set_to_none=True)

            if n_repeat > 1:
                if x.shape[0] % n_repeat != 0:
                    raise ValueError(f"Batch size {x.shape[0]} is not divisible by n_repeat={n_repeat}.")
                base_batch = x.shape[0] // n_repeat
                x_grouped = x.view(base_batch, n_repeat, *x.shape[1:])
                y = y.view(base_batch, n_repeat, *y.shape[1:])[:, 0]

                logits_sum = None
                for repeat_idx in range(n_repeat):
                    current_logits = model(x_grouped[:, repeat_idx])
                    logits_sum = current_logits if logits_sum is None else logits_sum + current_logits
                logits = logits_sum / n_repeat
            else:
                logits = model(x)

            loss = self.loss_function(logits, y)
            
            target_np = y.argmax(axis=1).cpu().numpy()
            pred_np = logits.argmax(axis=1).cpu().numpy()

            if opt is not None:
                loss.backward()
                opt.step()

            cm = confusion_matrix(target_np, pred_np, labels=range(len(self.classes)))
            cm_sum += cm
            cnt += 1
            loss_sum += float(loss.item())

            step = self.steps[mode]
            self._log_from_cm(cm, float(loss.item()), mode=mode, scope="batch", step=step) 

            accs = cm_sum.trace() / cm_sum.sum() * 100.0
            f1_micro = f1_score_from_confusion_matrix(cm_sum, macro=False)
            f1_macro = f1_score_from_confusion_matrix(cm_sum, macro=True)
            coehns_kappa = cohen_kappa_from_confusion_matrix(cm_sum)

            desc = f"{prefix:<12} {loss_sum/cnt:2.4f} acc {accs:2.3f} " \
                   f"f1 (mi/ma) {f1_micro:1.4f}/{f1_macro:1.4f} κ {coehns_kappa:2.3f}"
            
            self.steps[mode] += 1
            logger.progress_status(desc)
            logger.progress_advance(loader.batch_size)

        logger.progress_close()
        epoch_loss = loss_sum / max(cnt, 1) 
        self._log_from_cm(cm_sum, epoch_loss, mode=mode, scope="epoch", step=self.epoch_step)  
        logger.info(render_confusion_table_grid([format_confusion_table(self.classes, cm_sum)], header=f"{mode.upper()} confusion matrix", n_cols=1))

        return epoch_loss, cm_sum  
