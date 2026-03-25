from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

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

    def features(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        x = self.apply_preprocessors(x, len(self.preprocessors)+1)
        return self._features(x)

    def classifier(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        return self._classifier(x)

    @abstractmethod
    def feature_dim(self) -> int:  # pragma: no cover - abstract
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @abstractmethod
    def _features(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def _classifier(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        ...
