"""Per-call normalization along a chosen tensor dimension."""

import torch

from sleepwalker.models.preprocessors.Preprocessor import Preprocessor

class NormalizeAlongDim(Preprocessor):
    """Normalize by mean and variance along one tensor dimension.

    Args:
        dim: Dimension along which mean and variance are computed.
    """

    def __init__(self, dim = 0, channels=None):
        super().__init__(channels=channels)
        self.dim = dim

    def _update(self, data:torch.Tensor):
        """No-op warmup hook because this preprocessor is stateless."""
        ...

    def requires_warmup(self) -> bool:
        """Return whether this preprocessor requires warmup."""
        return False 

    def _transform(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor along the configured dimension."""
        return (data - data.mean(dim=self.dim, keepdim=True)) / (data.var(dim=self.dim, keepdim=True).sqrt() + 1e-6)
    
