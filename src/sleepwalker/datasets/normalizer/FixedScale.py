"""Configured amplitude scaling with no patient or window parameter fitting."""

import numpy as np

from sleepwalker.datasets.normalizer.base import Normalizer


class FixedScale(Normalizer):
    """Apply configured (values - mean) / std; neither parameter is fitted."""

    def __init__(self, mean=0, std=1):
        self.mean = float(mean)
        self.std = float(std)
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("FixedScale requires finite mean and positive finite std.")

    def __call__(self, values: np.ndarray, *, unit: str | None, is_recording: bool) -> tuple[np.ndarray, str | None]:
        return (np.asarray(values, dtype=float) - self.mean) / self.std, "dimensionless"
