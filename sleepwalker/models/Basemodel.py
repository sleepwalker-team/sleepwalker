from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import torch
from torch import nn
from torch.nn import ModuleList

class BaseModel(nn.Module, ABC):
    """Minimal base model that can own preprocessors.

    Preprocessors are simple callables with optional `requires_warmup()` and `update(x)`.
    They receive and return tensors shaped [B, T, C]. Warmup is orchestrated externally.
    """

    def __init__(self, preprocessors: Optional[Iterable] = None) -> None:
        super().__init__()
        self.preprocessors: ModuleList = ModuleList(preprocessors) if preprocessors is not None else ModuleList([])

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
