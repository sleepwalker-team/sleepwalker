"""Running mean/variance normalization preprocessor."""

import torch
from sleepwalker.models.preprocessors.Preprocessor import Preprocessor

class Normalize(Preprocessor):
    """Normalize features using running mean and variance estimates.

    The implementation maintains incremental statistics during warmup and then
    standardizes later inputs using those estimates.
    """

    def __init__(self, channels=None):
        super().__init__(channels=channels)
        self.register_buffer("mean", None)
        self.register_buffer("M2", None)
        self.count = 0
        # self.register_buffer("count", torch.tensor(0.0))

    def _update(self, data: torch.Tensor):
        """Update running mean and second-moment statistics from one batch."""
        # Batch statistics
        batch_count = data.shape[0]
        batch_mean = data.mean(dim=0)
        batch_M2 = ((data - batch_mean) ** 2).sum(dim=0)

        if self.mean is None:
            # Initialize with first batch
            self.mean = batch_mean
            self.M2 = batch_M2
            self.count = batch_count
        else:
            delta = batch_mean - self.mean
            total_count = self.count + batch_count

            # Update running mean and M2
            new_mean = self.mean + delta * batch_count / total_count
            new_M2 = self.M2 + batch_M2 + delta**2 * self.count * batch_count / total_count

            # Commit updates
            self.mean.copy_(new_mean)
            self.M2.copy_(new_M2)
            self.count = total_count

    def requires_warmup(self) -> bool:
        """Return whether this preprocessor requires warmup."""
        return True

    def _transform(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor using the accumulated running statistics."""
        if self.mean is not None and self.M2 is not None and self.count > 1:
            var = self.M2 / (self.count - 1)
            data = (data - self.mean) / (var.sqrt() + 1e-6)
        return data

    
