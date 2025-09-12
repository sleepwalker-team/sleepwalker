import os
from typing import Callable, Optional, Union, cast
import numpy as np
import torch
import tqdm
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel 
from abc import ABC, abstractmethod
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import OneCycleLR

from sleepwalker.models.base import BaseModel
from sleepwalker.utils import logger
from sleepwalker.datasets.base import BaseDataset, batch_collate

def unwrap_ddp(model: DistributedDataParallel | BaseModel) -> BaseModel:

    if isinstance(model, DistributedDataParallel):
        the_model = model.module if hasattr(model, "module") else model
        # Explicit type checking for pylance
        the_model = cast(BaseModel, the_model)
    else:
        the_model = model 

    return the_model

class Trainer(ABC):
    def __init__(
        self,
        epochs: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        batch_size: int = 32,
        lr: float = 1e-3,
        num_devices: int = 1,
        save_every: int = 1,
        num_workers: int = 16,
        backend: str = 'nccl',
        logger: Logger = Logger,
        experiment_name: str = 'dev',
        run_name: str = 'run',
        lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]] = None,
        early_stopping: Optional[int] = None
    ):
        self.epochs = epochs
        self.batch_size = batch_size // num_devices
        self.lr = lr
        self.num_devices = max(1, num_devices)
        self.save_every = save_every
        self.num_workers = num_workers 
        self.backend = backend
        self.logger = logger(experiment_name, run_name)
        self.early_stopping_patience = early_stopping

        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler
        

    @abstractmethod
    def train_step(self, model: BaseModel|DistributedDataParallel, batch) -> torch.Tensor:
        ...

    @abstractmethod
    def val_step(self, model: BaseModel|DistributedDataParallel, batch) -> torch.Tensor:
        ...

    @abstractmethod
    def warmup_step(self, model: BaseModel|DistributedDataParallel, batch):
        ...

    # single entry point
    def fit(self, model: BaseModel, train_ds: BaseDataset, val_ds: BaseDataset | None = None):
        use_multi = torch.cuda.is_available() and torch.cuda.device_count() > 1 and self.num_devices > 1
        if use_multi:
            print('Use ddp with', self.num_devices, 'devices')
            mp.set_start_method('spawn', force=True)
            world_size = min(self.num_devices, torch.cuda.device_count())
            mp.spawn(
                self._worker,
                args=(world_size, model, train_ds, val_ds),
                nprocs=world_size
            )
        else:
            device = self._select_device()
            print('Use single on device', device)
            self._worker(rank=0, world_size=1, model=model, train_ds=train_ds, val_ds=val_ds, override_device=device)

    # internal worker handles both single and ddp
    def _worker(self, rank:int, world_size:int, model: BaseModel, train_ds: BaseDataset, val_ds: BaseDataset|None, override_device: torch.device | None = None):
        
        # Set up model for either single training or DDP 
        is_ddp = world_size > 1
        if is_ddp:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
            torch.cuda.set_device(rank)
            dist.init_process_group(backend=self.backend, rank=rank, world_size=world_size)
            device = torch.device('cuda', index=rank)
            model = model.to(device)
            
            model: Union[BaseModel, DistributedDataParallel] = model  
            model = DistributedDataParallel(model, device_ids=[rank])
        else:
            device = override_device if override_device is not None else self._select_device()
            model = model.to(device)

        # Initialize optimizer and lr_scheduler
        opt = self.optimizer_fn(model)

        if self.lr_scheduler_fn is not None:
            lr_scheduler = self.lr_scheduler_fn(opt)
            if isinstance(lr_scheduler, OneCycleLR):
                raise ValueError(f"OneCycleLR is currently not supported") # TODO
        else:
            lr_scheduler = None

        # Create necessary loaders
        if is_ddp:
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if val_ds is not None else None
            shuffle_train, shuffle_val = False, False
        else:
            train_sampler = None
            val_sampler = None
            shuffle_train, shuffle_val = True, False

        pin = device.type == 'cuda'
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=shuffle_train, sampler=train_sampler,
            num_workers=self.num_workers, pin_memory=pin, collate_fn=batch_collate, drop_last=False,
            persistent_workers=True
        )
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=shuffle_val, sampler=val_sampler,
                num_workers=self.num_workers, pin_memory=pin, collate_fn=batch_collate, drop_last=False,
                persistent_workers=True
            )
        
        if self.early_stopping_patience and val_ds is None:
            logger.warning(f"early_stopping was set to true, but no validation dataset was given. Disabling early stopping")
            self.early_stopping_patience = None
        else:
            val_losses = []

        # Do the warmups
        warump_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=shuffle_train, sampler=train_sampler,
            num_workers=self.num_workers, pin_memory=pin, collate_fn=batch_collate, drop_last=False,
            persistent_workers=True
        )

        idx = 0
        the_model = unwrap_ddp(model) 

        for idx in range(len(the_model.preprocessors)):
            for batch in tqdm.tqdm(train_loader, desc=f'Warmup preprocessors [{idx}/{len(the_model.preprocessors) - 1}]', total=len(warump_loader), disable=rank != 0):
                batch["data"] = batch["data"].to(device, non_blocking=device.type == 'cuda')
                x = batch["data"]
                x = the_model.apply_preprocessors(batch["data"], idx+1)
                the_model.preprocessors[idx].update(x)

        for batch in tqdm.tqdm(train_loader, desc=f'Warmup trainer', total=len(warump_loader), disable=rank != 0):
            batch["data"] = batch["data"].to(device, non_blocking=device.type == 'cuda')
            batch["data"] = the_model.apply_preprocessors(batch["data"])
            self.warmup_step(the_model, batch)

        # Finally, start the training
        for epoch in range(self.epochs):
            model.train()
            if is_ddp:
                train_loader.sampler.set_epoch(epoch)

            train_loss_sum, n = torch.zeros(1, device=device), torch.zeros(1, device=device)
            n_train_batches = len(train_loader)
            for batch in tqdm.tqdm(train_loader, desc=f'Epoch [{epoch+1}/{self.epochs}]', total=n_train_batches, disable=rank != 0):
                batch["data"] = batch["data"].to(device, non_blocking=device.type == 'cuda')
                batch["target"] = batch["target"].to(device, non_blocking=device.type == 'cuda')
                opt.zero_grad(set_to_none=True)
                loss = self.train_step(model, batch)
                loss.backward()
                opt.step()

                train_loss_sum += loss.detach()
                n += 1

            if is_ddp:
                dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(n, op=dist.ReduceOp.SUM)
            if (not is_ddp) or (rank == 0):
                train_loss = (train_loss_sum / torch.clamp(n, min=1)).item()
                self.logger.log_metric('train/epoch_loss', train_loss, step=epoch)
                #print(f'epoch={epoch} train_loss={train_loss:.3f}')

            if lr_scheduler is not None:
                lr_scheduler.step()

            if val_loader is not None:
                model.eval()
                val_loss_sum, n = torch.zeros(1, device=device), torch.zeros(1, device=device)
                with torch.no_grad():
                    for batch in val_loader:
                        batch["data"] = batch["data"].to(device, non_blocking=device.type == 'cuda')
                        batch["target"] = batch["target"].to(device, non_blocking=device.type == 'cuda')
                        loss = self.val_step(model, batch)
                        loss = loss.detach()
                        val_loss_sum += loss
                        n += 1
                if is_ddp:
                    dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(n, op=dist.ReduceOp.SUM)
                if (not is_ddp) or (rank == 0):
                    val_loss = (val_loss_sum / torch.clamp(n, min=1)).item()
                    self.logger.log_metric('val/epoch_loss', val_loss, step=epoch)
                    #print(f'epoch={epoch} val_loss={val_loss:.3f}')

                # TODO REFACTOR THIS + BETTER METRIC LOGGING
                if self.early_stopping_patience:
                    val_losses.append(val_loss_sum)
                    imin = np.argmin(val_losses)
                    if len(val_losses) - imin > self.early_stopping_patience:
                        logger.info(f"Early stopping after {epoch} epochs - best epoch was {imin}") 
                        break

            # if (not is_ddp or rank == 0) and epoch % self.save_every == 0:
            #     to_save = model.module if hasattr(model, 'module') else model
            #     self._save(to_save, epoch)

        if is_ddp:
            dist.destroy_process_group()
        if (not is_ddp or rank == 0):
            self.logger.close()

    # helpers
    # def _save(self, model, epoch):
    #     path = f'checkpoint_epoch_{epoch}.pt'
    #     #torch.save(model.state_dict(), path)
    #     print(f'saved {path}')

    # def _to_device(self, batch:dict, device) -> dict:
    #     # if isinstance(batch, (list, tuple)):
    #     #     return tuple(self._to_device(x, device) for x in batch)
    #     if isinstance(batch, dict):
    #         return {k: v.to(device, non_blocking=device.type == 'cuda') for k, v in batch.items()}
    #     # if torch.is_tensor(batch):
    #     #     return batch.to(device, non_blocking=device.type == 'cuda')
    #     return batch

    def _select_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda', index=0)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device('mps')
        return torch.device('cpu')
