"""Fixed channel-wise tensor standardization."""

from collections.abc import Sequence

import torch
from torch import nn


class FixedChannelStandardizer(nn.Module):
    """Standardize the last tensor dimension with fixed channel statistics."""

    def __init__(self, means: Sequence[float], standard_deviations: Sequence[float]):
        super().__init__()
        means_tensor = torch.as_tensor(list(means), dtype=torch.float32)
        standard_deviations_tensor = torch.as_tensor(list(standard_deviations), dtype=torch.float32)
        if means_tensor.ndim != 1 or means_tensor.numel() == 0:
            raise ValueError("means must contain at least one channel value.")
        if standard_deviations_tensor.shape != means_tensor.shape:
            raise ValueError("means and standard_deviations must have identical shapes.")
        if not torch.isfinite(means_tensor).all():
            raise ValueError("means must be finite.")
        if not torch.isfinite(standard_deviations_tensor).all() or torch.any(standard_deviations_tensor <= 0):
            raise ValueError("standard_deviations must be finite and positive.")
        self.register_buffer("means", means_tensor)
        self.register_buffer("standard_deviations", standard_deviations_tensor)

    def requires_warmup(self) -> bool:
        """Return false because all statistics are fixed at construction."""
        return False

    def update(self, data: torch.Tensor) -> None:
        """Leave fixed statistics unchanged."""

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Standardize a tensor shaped with channels on its last dimension."""
        if data.ndim == 0 or data.shape[-1] != self.means.numel():
            raise ValueError(f"Expected {self.means.numel()} channels on the last dimension, got {tuple(data.shape)}.")
        shape = [1] * data.ndim
        shape[-1] = self.means.numel()
        return (data - self.means.view(shape)) / self.standard_deviations.view(shape)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Alias :meth:`transform` for module composition."""
        return self.transform(data)
