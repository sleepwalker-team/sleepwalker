"""Running mean/variance normalization preprocessor."""

import torch
from sleepwalker.models.preprocessors.Preprocessor import Preprocessor

class Normalize(Preprocessor):
    def __init__(self, stat_dims=None):
        """
        Normalize features using running mean and variance estimates.

        The implementation maintains incremental statistics during warmup and then
        standardizes later inputs using those estimates.
        Args:
            stat_dims: Dimensions to keep in the statistics (absolute indices).
                       E.g. stat_dims=(2, 3) on (B, N, F, D) → mean shape (F, D).
                       Default None: reduce over dim 0 only (mean shape = all non-batch dims).
        """
        super().__init__()
        self.register_buffer("mean", None)
        self.register_buffer("M2", None)
        self.count = 0
        self.stat_dims = stat_dims

    def _reduce_dims(self, ndim):
        if self.stat_dims is None:
            return (0,)
        stat = set(d % ndim for d in self.stat_dims)
        return tuple(d for d in range(ndim) if d not in stat)

    def _expand_for_broadcast(self, t, ndim):
        """Reshape t from stat_dims shape to ndim-dimensional shape with 1s at reduce dims."""
        if self.stat_dims is None:
            return t  # trailing dims right-align correctly without reshaping
        shape = [1] * ndim
        for i, d in enumerate(sorted(d % ndim for d in self.stat_dims)):
            shape[d] = t.shape[i]
        return t.reshape(shape)

    def update(self, data: torch.Tensor):
        """Update running mean and second-moment statistics from one batch."""
        reduce_dims = self._reduce_dims(data.dim())
        batch_count = 1
        for d in reduce_dims:
            batch_count *= data.shape[d]

        batch_mean = data.mean(dim=reduce_dims)
        batch_M2 = ((data - data.mean(dim=reduce_dims, keepdim=True)) ** 2).sum(dim=reduce_dims)

        if self.mean is None:
            self.mean = batch_mean
            self.M2 = batch_M2
            self.count = batch_count
        else:
            delta = batch_mean - self.mean
            total_count = self.count + batch_count

            new_mean = self.mean + delta * batch_count / total_count
            new_M2 = self.M2 + batch_M2 + delta**2 * self.count * batch_count / total_count

            self.mean.copy_(new_mean)
            self.M2.copy_(new_M2)
            self.count = total_count

    def requires_warmup(self) -> bool:
        """Return whether this preprocessor requires warmup."""
        return True

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor using the accumulated running statistics."""
        if self.mean is not None and self.M2 is not None and self.count > 1:
            var = self.M2 / (self.count - 1)
            mean = self._expand_for_broadcast(self.mean, data.dim())
            std = self._expand_for_broadcast(var.sqrt(), data.dim())
            data = (data - mean) / (std + 1e-6)
        return data
