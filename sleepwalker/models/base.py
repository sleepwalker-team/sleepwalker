from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """Minimal base model that can own preprocessors.

    Preprocessors are simple callables with optional `requires_warmup()` and `update(x)`.
    They receive and return tensors shaped [B, T, C]. Warmup is orchestrated externally.
    """

    def __init__(self, preprocessors: Optional[Iterable] = None) -> None:
        super().__init__()
        self.preprocessors: List = list(preprocessors) if preprocessors is not None else []

    def apply_preprocessors(self, x: torch.Tensor) -> torch.Tensor:
        for p in self.preprocessors:
            x = p(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.apply_preprocessors(x)
        return self._forward(x)

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        ...
