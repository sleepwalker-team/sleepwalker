from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional
from torch.utils.data import DataLoader
from sleepwalker.utils import logger

import torch
from torch import nn

def warmup_model(model: BaseModel, data_loader:DataLoader, device:str = "cuda") -> BaseModel:
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

class BaseModel(nn.Module, ABC):
    """Minimal base model that can own preprocessors.

    Preprocessors are simple callables with optional `requires_warmup()` and `update(x)`.
    They receive and return tensors shaped [B, T, C]. Warmup is orchestrated externally.
    """

    def __init__(self, preprocessors: Optional[Iterable] = None) -> None:
        super().__init__()
        self.preprocessors: List = list(preprocessors) if preprocessors is not None else []

    def apply_preprocessors(self, x: torch.Tensor, up:int = 0) -> torch.Tensor:
        if up < 0:
            up = 0
            
        for p in self.preprocessors[:up]:
            x = p(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.apply_preprocessors(x, len(self.preprocessors)+1)
        return self._forward(x)

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        ...
