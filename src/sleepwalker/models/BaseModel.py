"""Base model and model capability interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


class EmbeddingModel(ABC):
    """Capability for models that expose one embedding vector per sample."""

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Apply preprocessors and return embeddings."""
        return self.encode(self.apply_preprocessors(x))

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings for an already-preprocessed tensor."""
        ...

    @abstractmethod
    def feature_dim(self) -> int:
        ...


class ClassifierModel:
    """Marker for models whose normal ``nn.Module`` output is classification logits."""


class BaseModel(nn.Module):
    """Apply model preprocessors before one model-specific computation."""

    def __init__(self, preprocessors: Iterable[nn.Module] | None = None) -> None:
        super().__init__()
        self.preprocessors = nn.ModuleList([] if preprocessors is None else list(preprocessors))

    def apply_preprocessors(self, x: torch.Tensor, up: int | None = None) -> torch.Tensor:
        up = len(self.preprocessors) if up is None else up
        if up < 0 or up > len(self.preprocessors):
            raise ValueError(f"up must be between 0 and {len(self.preprocessors)}, got {up}.")
        for step in self.preprocessors[:up]:
            x = step.transform(x) if hasattr(step, "transform") else step(x)
        return x

    def forward(self, x: torch.Tensor):
        """Apply preprocessors exactly once and execute the model."""
        return self.compute(self.apply_preprocessors(x))

    def compute(self, x: torch.Tensor):
        """Execute the model on an already-preprocessed tensor."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement compute().")

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:  # pragma: no cover - abstract
        """Describe the positional input shape expected by ``forward``.

        Returns:
            A tuple ``(shape, meta)`` where ``shape`` is directly usable as a
            torchinfo-style input shape and ``meta`` carries semantic details
            such as layout and channel/time dimensions.

        Notes:
            ``shape`` should include the batch dimension and must describe what
            :meth:`forward` actually expects, which may differ from the raw
            dataset window when a model-specific adaptation is applied before
            feature extraction.

            ``meta`` should normally include at least:
            - ``layout``: tensor axis convention such as ``"BTC"``
            - ``ts_len``: time-axis length expected by the model
            - ``n_channels``: effective number of input channels
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement input_spec() before it can be packaged."
        )
