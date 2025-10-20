import inspect
import os
from typing import Callable, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from abc import ABC
from torch.optim.lr_scheduler import OneCycleLR

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.utils import logger

from sleepwalker.trainer.utils import cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix, store_checkpoint

def ensure_loss_signature(func: Callable, check_weight:bool = False) -> None:
    sig = inspect.signature(func)
    params = sig.parameters
    
    # 1) At least two parameters
    if len(params) < 2:
        raise TypeError(
            f"{func.__name__} must accept at least 2 parameters, "
            f"but only has {len(params)}"
        )
    
    if check_weight:
        # 2) Optional 'weight' argument
        weight_param = params.get("weight")
        if weight_param is None:
            raise TypeError(f"{func.__name__} must accept a 'weight' parameter")
        
        if weight_param.default is inspect._empty:
            raise TypeError(f"'weight' in {func.__name__} must be optional (have a default value)")

class MulticlassTrainer(ABC):
    def __init__(
        self,
        epochs: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        classes: list[str],
        loss_function: Callable,
        class_weights: Optional[dict[str, float]] = None,
        loss_mode: str = "regular",  # fixed (type: str, not Optional[str])
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

        self.loss_mode = loss_mode

        # Normalize class names once or keep original; here we keep original
        self.classes = classes
        self.num_classes = len(classes)  

        if self.loss_mode in ("inverse", "inverse-log"):  
            self.estimate_class_cnts = True
            self.class_cnts = torch.zeros(self.num_classes)  
        else:
            self.estimate_class_cnts = False

        if class_weights is not None:
            class_weights = {k.lower(): v for k, v in class_weights.items()}
            weight_list = []
            for c in classes:
                key = c.lower() 
                if key not in class_weights:
                    logger.warning(f"Did not find class weights for class {c}, assuming weight 1")
                    weight_list.append(1.0)
                else:
                    weight_list.append(class_weights[key])
            self.user_class_weights = torch.tensor(weight_list)
            self.class_weights = torch.tensor(weight_list)
        else:
            if self.loss_mode in ["inverse", "inverse-log"]:
                self.user_class_weights = torch.ones(self.num_classes)
                self.class_weights = torch.ones(self.num_classes)
            else:
                self.user_class_weights = None
                self.class_weights = None

        ensure_loss_signature(loss_function, self.class_weights is not None)
        self.loss_function = loss_function

    @staticmethod
    def target_to_multiclass(target, default_idx, min_event_seconds, raise_error=True):
        """
        Simplified: target is a 1D tensor of size (num_classes,).
        Returns a one-hot tensor based on threshold.
        """
        num_classes = target.shape[0]

        active = target > min_event_seconds
        active_sum = active.sum().item()

        if raise_error and active_sum > 1:
            raise ValueError("Multiple active classes found.")

        out = torch.zeros(num_classes, dtype=torch.float)

        if active_sum == 1:
            idx = active.nonzero(as_tuple=False).item()
            out[idx] = 1.0
        elif active_sum == 0:
            if default_idx is not None:
                out[default_idx] = 1.0
            elif raise_error:
                raise ValueError("Ambiguous class labels found with no active class and no default_idx.")

        return out

    @staticmethod
    def get_item(patient, time, data, target, target_extra = None, percentage:float = 0.5):
        try:
            freq = pd.to_timedelta(target.index.freq).total_seconds()
            targets = torch.tensor(target.sum().to_numpy())
            target = MulticlassTrainer.target_to_multiclass(targets, None, len(target)*freq*percentage, True)

            if target_extra is not None:
                freq = pd.to_timedelta(target_extra.index.freq).total_seconds()
                targets = torch.tensor(target_extra.sum().to_numpy())
                target_extra = MulticlassTrainer.target_to_multiclass(targets, None, len(target_extra)*freq*percentage, True)
            
            data = torch.from_numpy(data.values).float()
            return {"patient":patient, "time":time, "data":data, "target":target, "target_extra":target_extra}
        
        except Exception as e:
            pass
        return None

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Unified loss handling w/ optional class weights."""  
        target = y.argmax(dim=1)  
        if self.class_weights is None:
            return self.loss_function(logits, target)  
        
        w = self.class_weights.to(logits.device)  
        if torch.unique(w).numel() <= 1:  
            return self.loss_function(logits, target)  
        return self.loss_function(logits, target, weight=w)  

    def _log_from_cm(self, cm: np.ndarray, loss_value: float, mode: str, scope: str = "batch"):
        """Centralized metric logging from confusion matrix."""  
        total = cm.sum()  
        if total == 0:  
            return  
        
        acc = cm.trace() / total * 100.0  
        f1_micro = f1_score_from_confusion_matrix(cm, macro=False)  
        f1_macro = f1_score_from_confusion_matrix(cm, macro=True)  
        kappa = cohen_kappa_from_confusion_matrix(cm)  
        logger.metric(f"{scope}/{mode}/accuracy", acc)  
        logger.metric(f"{scope}/{mode}/f1_micro", f1_micro)  
        logger.metric(f"{scope}/{mode}/f1_macro", f1_macro)  
        logger.metric(f"{scope}/{mode}/coehns_kappa", kappa)  
        logger.metric(f"{scope}/{mode}/loss", float(loss_value)) 

    def warmup_preprocessors(self, model: BaseModel, data_loader:DataLoader, device:str = "cuda") -> BaseModel:
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
                # No warmup required -> Set tqdm bar to final value directly                
                logger.progress_advance(total_batches*batch_size)
            logger.progress_close()

        return model

    def _warmup(self, model, data_loader, device = "cuda"):
        logger.context("Warmup preprocessors")
        self.warmup_preprocessors(model, data_loader, device) 
        logger.uncontext()

        if self.estimate_class_cnts:
            logger.progress_start(len(data_loader)*data_loader.batch_size, desc="Warmup weights", leave=True)

            for batch in data_loader:
                y = batch["target"].to(device)
                target = y.argmax(dim=1)
                idx, cnt = torch.unique(target, return_counts=True)
                
                self.class_cnts = self.class_cnts.to(y.device)
                self.class_weights = self.class_weights.to(y.device) # type: ignore
                self.user_class_weights = self.user_class_weights.to(y.device) # type: ignore

                self.class_cnts[idx] += cnt
                if self.loss_mode == "inverse":
                    # Weight classes by their (inverse) occurrence. This can lead to relatively small losses,
                    # hence we will also weight normalize it. This is technically not necessary.
                    self.class_weights = self.user_class_weights * torch.clamp(1.0 / self.class_cnts, min = 1e-4)
                    self.class_weights /= self.class_weights.sum() 
                else:
                    """
                    See 
                        - MRASleepNet: a multi-resolution attention network for sleep stage classification using single-channel EEG by Rui Yu, Zhuhuang Zhou, Shuicai Wu, Xiaorong Gao and Guangyu Bin in Journal of Neural Engineering 2022, https://github.com/YuRui8879/MRASleepNet/blob/781aee2d2ff1422c598b099081a4c1d7d4bd05d7/DataAdapter/DataAdapter.py#L110
                        - An Attention-Based Deep Learning Approach for Sleep Stage Classification With Single-Channel EEG by Eldele et al. in IEEE TRANSACTIONS ON NEURAL SYSTEMS AND REHABILITATION ENGINEERING 2021, https://github.com/emadeldeen24/AttnSleep/blob/6b4d2665884628c8a7bb09f36589a8ec0992f8e2/utils/util.py#L62
                    """
                    total = self.class_cnts.sum()
                    factor = 1.0 / self.class_cnts.shape[0]
                    mu = factor * self.user_class_weights
                    self.class_weights = mu * torch.clamp(torch.log( (total * mu) / self.class_cnts), min=1.0)
                logger.progress_advance(data_loader.batch_size)
            logger.progress_close()

        return model

    def apply_model(self, model, x, is_test:bool=False):
        return model(x)

    def run_epoch(self, loader, opt, model, prefix=""):
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)
        nc = self.num_classes
        
        loss_sum = 0
        cm_sum = np.zeros((nc, nc), dtype=np.int64)
        cnt = 0

        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"

        for batch in loader:
            x = batch["data"].to(self.device)
            y = batch["target"].to(self.device)
            
            if opt is not None: opt.zero_grad(set_to_none=True)
            
            logits = self.apply_model(model, x, is_test=mode=="TEST") # Typically model(x) would be enough here, but this way we can override self.apply_model later :-)
            loss = self._loss(logits, y)
            
            target_np = y.argmax(axis=1).cpu().numpy()
            pred_np = logits.argmax(axis=1).cpu().numpy()

            if opt is not None:
                loss.backward()
                opt.step()

            cm = confusion_matrix(target_np, pred_np, labels=range(len(self.classes)))
            cm_sum += cm
            cnt += 1
            loss_sum += float(loss.item())

            self._log_from_cm(cm, float(loss.item()), mode=mode, scope="batch") 

            accs = cm_sum.trace() / cm_sum.sum() * 100.0
            f1_micro = f1_score_from_confusion_matrix(cm_sum, macro=False)
            f1_macro = f1_score_from_confusion_matrix(cm_sum, macro=True)
            coehns_kappa = cohen_kappa_from_confusion_matrix(cm_sum)

            desc = f"{prefix:<12} {loss_sum/cnt:2.4f} acc {accs:2.3f} " \
                   f"f1 (mi/ma) {f1_micro:1.4f}/{f1_macro:1.4f} κ {coehns_kappa:2.3f}"
            
            logger.progress_status(desc)
            logger.progress_advance(loader.batch_size)

        logger.progress_close()
        epoch_loss = loss_sum / max(cnt, 1) 
        self._log_from_cm(cm_sum, epoch_loss, mode=mode, scope="epoch")  

        return epoch_loss, cm_sum  
    
    def test(self, model: BaseModel, test_loader):
        model.eval()
        with torch.inference_mode():
            test_loss, test_cm = self.run_epoch(test_loader, None, model, f"TEST") 
        return test_loss, test_cm
    
    def fit(self, model: BaseModel, train_loader, val_loader = None):
        model = model.to(self.device)
        opt = self.optimizer_fn(model)

        if self.lr_scheduler_fn is not None:
            lr_scheduler = self.lr_scheduler_fn(opt)
            if isinstance(lr_scheduler, OneCycleLR):
                raise ValueError(f"OneCycleLR is currently not supported") # TODO
        else:
            lr_scheduler = None

        if self.early_stopping_patience and val_loader is None:
            logger.warning(f"early_stopping was set to true, but no validation dataset was given. Disabling early stopping")
            self.early_stopping_patience = None

        model = self._warmup(model, train_loader, self.warmup_device)
        val_losses: list[float] = []
        losses = []
        cms = []

        for epoch in range(self.epochs):
            model.train()
            loss, cm = self.run_epoch(train_loader, opt, model, f"TRAIN [{epoch+1}/{self.epochs}]")
            cms.append({"train":cm})
            losses.append({"train":loss})

            if self.save_every > 0 and (epoch % self.save_every == 0):
                logger.info(f"Logging intermediate model after {epoch} epochs.")
                
                folder = store_checkpoint(model, opt, lr_scheduler)
                logger.artifact(path=os.path.join(folder, "model.pt"), dest=f"{epoch}")
                logger.artifact(path=os.path.join(folder, "optimizer.pt"), dest=f"{epoch}")
                if lr_scheduler:
                    logger.artifact(path=os.path.join(folder, "scheduler.pt"), dest=f"{epoch}")

            if lr_scheduler is not None:
                lr_scheduler.step()

            if val_loader is not None:
                model.eval()
                with torch.inference_mode():
                    val_loss, val_cm = self.run_epoch(val_loader, opt, model, f"VAL [{epoch+1}/{self.epochs}]") 
                cms.append({"val": val_cm}) 
                losses.append({"val": val_loss}) 
                val_losses.append(val_loss) 
            
                imin = np.argmin(val_losses)
                if self.early_stopping_patience and (len(val_losses) - imin >= self.early_stopping_patience):
                    logger.info(f"Early stopping after {epoch} epochs - best epoch was {imin}") 
                    break
        
        return losses, cms