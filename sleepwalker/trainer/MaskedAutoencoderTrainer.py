
import os
import shutil
import tempfile
from typing import Callable, Optional
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import DataLoader
from abc import ABC

from sleepwalker.utils import logger
from sleepwalker.trainer.BaseTrainer import build_lr_scheduler
from sleepwalker.trainer.utils import cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix, store_checkpoint

def masked_mse(y_pred, y_true, mask):
    sq_err = (y_pred - y_true)**2
    mask_expanded = mask.expand_as(sq_err)
    return (sq_err * mask_expanded).sum() / mask_expanded.sum().clamp(min=1)

def masked_mean(feats, channel_mask):
    while channel_mask.dim() < feats.dim():
        channel_mask = channel_mask.unsqueeze(1)
    mask = channel_mask.to(feats.dtype)
    return (feats * mask).sum(-1) / mask.sum(-1).clamp(min=1)

def center_per_patient(X, patient_ids):
    unique_ids = torch.unique(patient_ids)
    for uid in unique_ids:
        mask = patient_ids == uid
        X[mask] -= X[mask].mean(axis=0)
    return X

class MaskedAutoencoderTrainer(ABC):
    def __init__(
        self,
        epochs: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        classes: list[str],
        device: str = "cuda:0",
        warmup_device: str = "cpu",
        save_every: int = 1,
        downstream_tasks = None,
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
        early_stopping: Optional[int] = None,
        groups=None,
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
        if downstream_tasks is None:
            self.downstream_tasks = {}
        else:
            self.downstream_tasks = downstream_tasks 

        self.groups = groups

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
        if scope == 'epoch':
            print(f'{step} | {mode} kappa {kappa:.3f}   f1_macro {f1_macro:.3f}')

    def _log_loss(self, loss, mode, scope='batch', step=0):
        logger.metric(f"{scope}/{mode}/loss", loss, step=step) 

    def warmup_preprocessors(self, model: torch.nn.Module, data_loader:DataLoader, device:str = "cuda") -> torch.nn.Module:
        model.to(device)
        total_batches = len(data_loader)
        batch_size = data_loader.batch_size  

        if batch_size is None:
            raise ValueError(f"batch_size should not be None here.")

        logger.progress_start(total_batches*batch_size, leave=True)
        for batch in data_loader:
            for modality in self.groups:
                for idx in range(len(model.preprocessors[modality])):
                    if model.preprocessors[modality][idx].requires_warmup():
                            x = batch[f'data_{modality}'] 
                            x = model.apply_preprocessors(x, modality, idx)
                            model.preprocessors[modality][idx].update(x)
            logger.progress_advance(batch_size)
        logger.progress_close()

        return model

    def run_epoch(self, loader, opt, model, prefix="", lr_scheduler=None):
        logger.progress_start(total=len(loader) * loader.batch_size, desc=prefix, leave=True)
        
        loss_sum = 0
        cnt = 0

        mode = "train" if "TRAIN" in prefix else "val" if "VAL" in prefix else "test"

        for batch in loader:
            if opt is not None:
                opt.zero_grad(set_to_none=True)

            y_true = []
            y_pred = []
            mask = []
            for modality in self.groups:
                _x = batch[f'data_{modality}'].to(self.device)
                channel_mask = batch[f'mask_{modality}'].to(self.device)
                _x = model.apply_preprocessors(_x, modality)
                _y_pred, _mask = model(_x, modality)
                # Mask the output mask with whatever channels are present
                _mask = _mask * channel_mask[:, None, None, :]

                y_pred.append(_y_pred)
                mask.append(_mask)
                y_true.append(_x.clone())

            y_true = torch.cat(y_true, -1)
            y_pred = torch.cat(y_pred, -1)
            mask = torch.cat(mask, -1)
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
            if lr_scheduler is not None:
                lr_scheduler.step()

        logger.progress_close()
        epoch_loss = loss_sum / max(cnt, 1) 
        self._log_loss(epoch_loss, mode=mode, scope='epoch', step=self.epoch_step)

        return epoch_loss

    @torch.inference_mode
    def run_downstream_tasks(self, val_loader, test_loader, model):
        model.eval()
        logger.progress_start(total=len(val_loader) * val_loader.batch_size, desc='Embed VAL', leave=True)

        # Embed the training data for the downstream model
        X_train = []
        y_train = {k: [] for k in self.downstream_tasks.keys()}
        patient_ids = []
        for batch in val_loader:
            pids = batch['patient']
            embeddings = {}
            for modality in self.groups:
                _x = batch[f'data_{modality}'].to(self.device)
                channel_mask = batch[f'mask_{modality}'].to(self.device)  # (B, C)
                _x = model.apply_preprocessors(_x, modality)
                feats = model.encode(_x, modality)
                embeddings[modality] = masked_mean(feats, channel_mask)

            X_train.append(embeddings)
            for k in self.downstream_tasks.keys():
                y_train[k].append(batch[f'target_{k}'].argmax(-1))
            patient_ids.append(pids)
            logger.progress_advance(val_loader.batch_size)

        logger.progress_close()

        X_train = {k: torch.cat([d[k] for d in X_train], 0) for k in X_train[0]}
        y_train = {k: torch.cat(v, 0) for k, v in y_train.items()}
        patient_ids = torch.cat(patient_ids, 0)
        X_train = {k: center_per_patient(X_train[k], patient_ids) for k in X_train}

        # Embed the test data for the downstream model
        logger.progress_start(total=len(test_loader) * test_loader.batch_size, desc='Embed TEST', leave=True)
        X_test = []
        y_test = {k: [] for k in self.downstream_tasks.keys()}
        patient_ids = []
        for batch in test_loader:
            pids = batch['patient']
            embeddings = {}
            for modality in self.groups:
                _x = batch[f'data_{modality}'].to(self.device)
                channel_mask = batch[f'mask_{modality}'].to(self.device)  # (B, C)
                _x = model.apply_preprocessors(_x, modality)
                feats = model.encode(_x, modality)
                embeddings[modality] = masked_mean(feats, channel_mask)

            X_test.append(embeddings)
            for k in self.downstream_tasks.keys():
                y_test[k].append(batch[f'target_{k}'].argmax(-1))
            patient_ids.append(pids)
            logger.progress_advance(test_loader.batch_size)

        logger.progress_close()

        X_test = {k: torch.cat([d[k] for d in X_test], 0) for k in X_test[0]}
        y_test = {k: torch.cat(v, 0) for k, v in y_test.items()}
        patient_ids = torch.cat(patient_ids, 0)
        X_test = {k: center_per_patient(X_test[k], patient_ids) for k in X_test}

        for k, cfg in self.downstream_tasks.items():
            middle_slices = cfg['n_slices']
            # Slice out middle fo prediction
            T = X_train[cfg['embeddings'][0]].shape[1]
            _from = T//2 - middle_slices // 2
            _to = T//2 + middle_slices // 2

            X_train_embeddings = np.concatenate([X_train[emb_key][:, _from:_to].mean(1).cpu().numpy() for emb_key in self.downstream_tasks[k]['embeddings']], -1)
            X_test_embeddings = np.concatenate([X_test[emb_key][:, _from:_to].mean(1).cpu().numpy() for emb_key in self.downstream_tasks[k]['embeddings']], -1)
            self.run_classification(X_train_embeddings, X_test_embeddings, y_train[k], y_test[k], cfg, label=k)

    @torch.inference_mode
    def run_classification(self, X_train, X_test, y_train, y_test, cfg, label='sleep'):
        if len(cfg['class_weights']) > 0:
            clf = LogisticRegression(max_iter=10000, class_weight=cfg['class_weights'])
        else:
            clf = LogisticRegression(max_iter=10000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        self._log_from_cm(cm, mode=label, scope='epoch', step=self.epoch_step)
    
    def fit(self, model: torch.nn.Module, train_loader, val_loader=None, test_loader=None, warmup_loader=None):
        opt = self.optimizer_fn(model)

        lr_scheduler, scheduler_per_batch = build_lr_scheduler(self.lr_scheduler_fn, opt, self.epochs, len(train_loader))

        if self.early_stopping_patience and val_loader is None:
            logger.warning(f"early_stopping was set to true, but no validation dataset was given. Disabling early stopping")
            self.early_stopping_patience = None

        logger.context("Warmup preprocessors")
        if warmup_loader is None:
            warmup_loader = train_loader
        self.warmup_preprocessors(model, warmup_loader, self.warmup_device) 
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
            loss = self.run_epoch(train_loader, opt, model, f"TRAIN [{epoch+1}/{self.epochs}]", lr_scheduler if scheduler_per_batch else None)
            losses.append({"train":loss})

            if self.save_every > 0 and (epoch % self.save_every == 0):
                logger.info(f"Logging intermediate model after {epoch} epochs.")
                
                self.last_folder = store_checkpoint(model, opt, lr_scheduler, tempfile.mkdtemp(prefix=f"checkpoint_epoch_{epoch}_"))
                logger.artifact(path=os.path.join(self.last_folder, "model.pt"), dest=f"{epoch}")
                logger.artifact(path=os.path.join(self.last_folder, "optimizer.pt"), dest=f"{epoch}")
                if lr_scheduler:
                    logger.artifact(path=os.path.join(self.last_folder, "scheduler.pt"), dest=f"{epoch}")

            if lr_scheduler is not None and not scheduler_per_batch:
                lr_scheduler.step()

            if val_loader is not None:
                model.eval()
                with torch.inference_mode():
                    val_loss = self.run_epoch(val_loader, None, model, f"VAL [{epoch+1}/{self.epochs}]") 
                losses[-1]["val"] =  val_loss
                val_losses.append(val_loss)

                if test_loader is not None:
                    self.run_downstream_tasks(val_loader, test_loader, model)
            
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
