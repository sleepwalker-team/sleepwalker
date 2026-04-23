"""Shared model base class with optional preprocessor support.

This module defines the minimal interface expected by the repository's trainer
and meta-model layers. Concrete architectures subclass ``BaseModel`` and
implement feature extraction plus a classifier head, while optional
preprocessors are warmed externally by trainer code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import ModuleList

class BaseModel(nn.Module, ABC):
    """Base class for Sleepwalker models.

    The class assumes a common tensor layout of ``[B, T, C]`` and provides a
    small amount of structure around optional preprocessors. Tests in
    ``tests/test_metamodel.py`` and ``tests/test_deployment.py`` exercise this
    interface through lightweight dummy implementations.

    Args:
        preprocessors: Optional iterable of modules applied before
            feature extraction. Preprocessors may implement
            ``requires_warmup()`` and ``update(x)``; warmup is orchestrated by
            trainer code rather than by the model itself.
    """

    def __init__(self, preprocessors: Optional[Iterable] = None) -> None:
        super().__init__()
        self.preprocessors: ModuleList = ModuleList(preprocessors) if preprocessors is not None else ModuleList([])

    def apply_preprocessors(self, x: torch.Tensor, up:int = 0) -> torch.Tensor:
        """Apply preprocessors up to a given slice boundary.

        Args:
            x: Input tensor shaped ``[B, T, C]``.
            up: Exclusive upper bound on the preprocessors to apply. Values
                below zero are clamped to zero.

        Returns:
            The transformed tensor.
        """
        if up < 0:
            up = 0
            
        for p in self.preprocessors[:up]:
            x = p.transform(x) if hasattr(p, "transform") else p(x)
        return x

    def features(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        """Run preprocessors and then compute the latent feature embedding."""
        x = self.apply_preprocessors(x, len(self.preprocessors)+1)
        return self._features(x)

    def classifier(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        """Apply the task head to an already-computed embedding."""
        return self._classifier(x)

    @abstractmethod
    def feature_dim(self) -> int:  # pragma: no cover - abstract
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full model from raw inputs to logits or task outputs."""
        return self.classifier(self.features(x))

    @abstractmethod
    def _features(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def _classifier(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        ...
