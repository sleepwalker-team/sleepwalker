"""Scalers fitted once on one recording and applied unchanged to its windows."""

import numpy as np

from sleepwalker.datasets.normalizer.base import Normalizer


class RecordingZScore(Normalizer):
    """Fit one EDF/logical-channel/alias mean and population standard deviation."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, values: np.ndarray, *, unit: str | None, is_recording: bool):
        """Learn recording statistics; return None for unusable input."""
        if not is_recording:
            raise ValueError("Recording scalers must fit on a complete recording.")
        values = np.asarray(values, dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            return None
        self.mean = float(np.mean(values))
        self.std = float(np.std(values))
        if np.ptp(values) == 0 or not np.isfinite(self.std) or self.std <= 0:
            return None
        return self

    def __call__(self, values: np.ndarray, *, unit: str | None, is_recording: bool) -> tuple[np.ndarray, str | None]:
        if self.mean is None or self.std is None:
            raise ValueError("RecordingZScore must be fitted during recording preparation.")
        return (np.asarray(values, dtype=float) - self.mean) / self.std, "dimensionless"


class RecordingRobustScale(Normalizer):
    """Fit one recording median and configured quantile span (default IQR)."""

    def __init__(self, quantiles=(0.25, 0.75)):
        self.quantiles = tuple(float(value) for value in quantiles)
        if len(self.quantiles) != 2 or not 0 <= self.quantiles[0] < self.quantiles[1] <= 1:
            raise ValueError("quantiles must be two increasing probabilities in [0, 1].")
        self.median = None
        self.scale = None

    def fit(self, values: np.ndarray, *, unit: str | None, is_recording: bool):
        """Learn recording statistics; return None for unusable input."""
        if not is_recording:
            raise ValueError("Recording scalers must fit on a complete recording.")
        values = np.asarray(values, dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            return None
        self.median = float(np.median(values))
        low, high = np.quantile(values, self.quantiles)
        self.scale = float(high - low)
        if not np.isfinite(self.scale) or self.scale <= 0:
            return None
        return self

    def __call__(self, values: np.ndarray, *, unit: str | None, is_recording: bool) -> tuple[np.ndarray, str | None]:
        if self.median is None or self.scale is None:
            raise ValueError("RecordingRobustScale must be fitted during recording preparation.")
        return (np.asarray(values, dtype=float) - self.median) / self.scale, "dimensionless"
