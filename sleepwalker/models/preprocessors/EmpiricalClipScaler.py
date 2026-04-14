"""Quantile-based clipping and min-max scaling preprocessor."""

import torch

from sleepwalker.models.preprocessors.Preprocessor import Preprocessor
from sleepwalker.utils import logger

class EmpiricalClipScaler(Preprocessor):
    """Clip features to empirical quantile bounds and scale to `[0, 1]`.

    Args:
        q: Upper quantile used to estimate clipping bounds. The lower quantile
            is `1 - q`.
        scale: Additional multiplicative expansion applied to the empirical
            bounds.
    """
        
    def __init__(self, q=0.9, scale=1, **kwargs):
        super().__init__()
        self.mins = None
        self.maxs = None
        self.q = q
        self.scale = scale

    def requires_warmup(self) -> bool:
        """Return whether this preprocessor requires warmup."""
        return True

    def update(self, data: torch.Tensor):
        """Update empirical min/max bounds from one batch."""
        x = data
        (_, _, n_features) = x.shape
        _x = x.reshape(-1, n_features)

        if self.mins is None:
            self.mins = (torch.ones((n_features))*1e10).to(x.device)
        if self.maxs is None:
            self.maxs = (torch.ones((n_features))*1e-10).to(x.device)

        emp_mins = torch.quantile(_x, dim=0, q=1-self.q) * 1/self.scale
        emp_maxs = torch.quantile(_x, dim=0, q=self.q) * self.scale
        self.mins = torch.minimum(emp_mins, self.mins)
        self.maxs = torch.maximum(emp_maxs, self.maxs)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Clamp and rescale a tensor using the learned bounds."""
        if self.mins is None or self.maxs is None:
            logger.warning('EmpiricalClipScaler is not fitted; Return data as-is')
            return data
        
        # Clamp data to [min, max] per channel
        clamped = torch.clamp(data, min=self.mins, max=self.maxs)

        # Avoid division by zero
        denom = self.maxs - self.mins
        denom[denom == 0] = 1.0

        # Scale to [0, 1]
        scaled = (clamped - self.mins) / denom

        return scaled
        
