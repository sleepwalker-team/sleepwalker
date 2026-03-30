
import os
import shutil
import tempfile
from typing import Callable, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from abc import ABC
from torch.optim.lr_scheduler import OneCycleLR

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.utils import logger
from sleepwalker.trainer.utils import cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix, store_checkpoint

def masked_mse(y_pred, y_true, mask):
    sq_err = (y_pred - y_true)**2  # [B, F, N, D]
    mask_expanded = mask.expand_as(sq_err) 
    return sq_err[mask_expanded.bool()].mean()

class MaskedAutoencoderTrainer(ABC):
    def __init__(
        self,
        epochs: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        classes: list[str],
        device: str = "cuda:0",
        warmup_device: str = "cpu",
        save_every: int = 1,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
        early_stopping: Optional[int] = None,
    ):
        self.epochs = epochs
        self.save_every = save_every
        self.early_stopping_patience = early_stopping
        self.device = device
        self.warmup_device = warmup_device
        
        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler

        # Normalize class names once or keep original; here we keep original
        self.classes = classes
        self.num_classes = len(classes)  
        
        self.loss_function = masked_mse

    def _log_from_cm(self, cm: np.ndarray, mode: str, scope: str = "batch", step:int = 0):
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

    def _log_loss(self, loss, mode, scope='batch', step=0):
        logger.metric(f"{scope}/{mode}/loss", loss, step=step) 

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

    def run_epoch(self, loader, opt, model, prefix=""):
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)
        
        loss_sum = 0
        cnt = 0

        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"

        for batch in loader:
            x = batch['data'].to(self.device)
            #class_label = batch['target'].to(self.device)
            y_true = model.apply_preprocessors(x, len(model.preprocessors)+1)

            if opt is not None:
                opt.zero_grad(set_to_none=True)

            y_pred, mask = model(x)
            loss = self.loss_function(y_pred, y_true, mask)
            
            if opt is not None:
                loss.backward()
                opt.step()

            cnt += 1
            curr_loss = float(loss.item())
            loss_sum += curr_loss

            step = self.steps[mode]
            self._log_loss(curr_loss, mode=mode, scope='batch', step=step)
            
            desc = f"{prefix:<12} {loss_sum/cnt:2.4f}"
            
            self.steps[mode] += 1
            logger.progress_status(desc)
            logger.progress_advance(loader.batch_size)

        logger.progress_close()
        epoch_loss = loss_sum / max(cnt, 1) 
        self._log_loss(epoch_loss, mode=mode, scope='epoch', step=self.epoch_step)

        return epoch_loss

    @torch.inference_mode
    def run_sleep_classification(self, val_loader, test_loader, model):
        model.eval()
        logger.progress_start(total=len(val_loader) * val_loader.batch_size, desc='Embed VAL', leave=True)

        # Embed the training data for the downstream model
        X = []
        y = []
        for batch in val_loader:
            x = batch['data'].to(self.device)
            class_label = batch['target'].argmax(-1).to(self.device)
            embeddings = model.embed(x)

            X.append(embeddings)
            y.append(class_label)
            logger.progress_advance(val_loader.batch_size)

        logger.progress_close()

        X_train = torch.cat(X, 0).cpu().numpy()
        y_train = torch.cat(y, 0).cpu().numpy()

        # Embed the test data for the downstream model
        logger.progress_start(total=len(test_loader) * test_loader.batch_size, desc='Embed TEST', leave=True)
        X = []
        y = []
        for batch in test_loader:
            x = batch['data'].to(self.device)
            class_label = batch['target'].argmax(-1).to(self.device)
            embeddings = model.embed(x)

            X.append(embeddings)
            y.append(class_label)
            logger.progress_advance(test_loader.batch_size)

        logger.progress_close()

        X_test = torch.cat(X, 0).cpu().numpy()
        y_test = torch.cat(y, 0).cpu().numpy()

        # Train three models:
        #   1. Take the average window tokens (all of them)
        #   2. Take the average of the middle 150 seconds
        #   3. Take the average of the middle 30 seconds (since they have context already)

        X_train_sub150 = X_train[:, 31:81].mean(1).mean(-1)
        X_train_sub30 = X_train[:, 51:61].mean(1).mean(-1)

        X_test_sub150 = X_test[:, 31:81].mean(1).mean(-1)
        X_test_sub30 = X_test[:, 51:61].mean(1).mean(-1)

        clf_sub = LogisticRegression(max_iter=10000)
        clf_sub.fit(X_train_sub150, y_train)
        y_pred = clf_sub.predict(X_test_sub150)
        cm = confusion_matrix(y_test, y_pred)
        self._log_from_cm(cm, mode='SUB150', scope='epoch', step=self.epoch_step)

        clf_sub = LogisticRegression(max_iter=10000)
        clf_sub.fit(X_train_sub30, y_train)
        y_pred = clf_sub.predict(X_test_sub30)
        cm = confusion_matrix(y_test, y_pred)
        self._log_from_cm(cm, mode='SUB30', scope='epoch', step=self.epoch_step)
    
    def fit(self, model: BaseModel, train_loader, val_loader=None, test_loader=None):
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

        logger.context("Warmup preprocessors")
        self.warmup_preprocessors(model, train_loader, self.warmup_device) 
        logger.uncontext()

        model = model.to(self.device)
        val_losses: list[float] = []
        losses = []

        self.best_model_idx = None
        self.best_checkpoint = None
        self.steps = {"train":0, "val":0, "test":0}
        self.epoch_step = 0
        self.last_folder = None

        for epoch in range(self.epochs):
            model.train()
            loss = self.run_epoch(train_loader, opt, model, f"TRAIN [{epoch+1}/{self.epochs}]")
            losses.append({"train":loss})

            if self.save_every > 0 and (epoch % self.save_every == 0):
                logger.info(f"Logging intermediate model after {epoch} epochs.")
                
                self.last_folder = store_checkpoint(model, opt, lr_scheduler, tempfile.mkdtemp(prefix=f"checkpoint_epoch_{epoch}_"))
                logger.artifact(path=os.path.join(self.last_folder, "model.pt"), dest=f"{epoch}")
                logger.artifact(path=os.path.join(self.last_folder, "optimizer.pt"), dest=f"{epoch}")
                if lr_scheduler:
                    logger.artifact(path=os.path.join(self.last_folder, "scheduler.pt"), dest=f"{epoch}")

            if lr_scheduler is not None:
                lr_scheduler.step()

            if val_loader is not None:
                model.eval()
                with torch.inference_mode():
                    val_loss = self.run_epoch(val_loader, None, model, f"VAL [{epoch+1}/{self.epochs}]") 
                losses[-1]["val"] =  val_loss
                val_losses.append(val_loss)

                if test_loader is not None:
                    cm = self.run_sleep_classification(val_loader, test_loader, model)
            
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
                        "best_model":imin,
                        "checkpoint":self.best_checkpoint
                    }
            
            self.epoch_step += 1

        if self.last_folder is not None:
            return { "losses":losses, "checkpoint":self.last_folder}
        else:
            return { "losses":losses}
