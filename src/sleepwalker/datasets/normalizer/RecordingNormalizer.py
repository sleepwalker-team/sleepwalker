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


class RecordingMeanTrimmedZScore(Normalizer):
    """Scale with statistics of samples closest to the full-recording mean.

    Retain floor(retained_fraction * N) samples (default 95%). Ties at the
    cutoff are resolved in recording order, keeping exactly that count. Fit
    their mean and sample standard deviation (ddof=1); do not refit on windows.
    Insufficient samples, non-finite recordings and zero variance reject the
    recording by returning None, as for the other recording scalers.
    """

    def __init__(self, retained_fraction: float = 0.95):
        self.retained_fraction = float(retained_fraction)
        if not np.isfinite(self.retained_fraction) or not 0 < self.retained_fraction <= 1:
            raise ValueError("retained_fraction must be finite and in (0, 1].")
        self.mean = None
        self.std = None

    def fit(self, values: np.ndarray, *, unit: str | None, is_recording: bool):
        """Fit one complete logical source/physical alias; clear stale state."""
        if not is_recording:
            raise ValueError("Recording scalers must fit on a complete recording.")
        self.mean = None
        self.std = None
        values = np.asarray(values, dtype=float)
        if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 1):
            raise ValueError("RecordingMeanTrimmedZScore expects one channel shaped [N] or [N, 1].")
        values = values.reshape(-1)
        count = int(np.floor(self.retained_fraction * values.size))
        if count < 2 or not np.isfinite(values).all():
            return None
        recording_mean = float(np.mean(values))
        if not np.isfinite(recording_mean):
            return None
        distances = np.abs(values - recording_mean)
        cutoff = np.partition(distances, count - 1)[count - 1]
        keep = distances < cutoff
        ties = np.flatnonzero(distances == cutoff)
        keep[ties[:count - np.count_nonzero(keep)]] = True
        retained = values[keep]
        mean = float(np.mean(retained))
        std = float(np.std(retained, ddof=1))
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            return None
        self.mean, self.std = mean, std
        return self

    def __call__(self, values: np.ndarray, *, unit: str | None, is_recording: bool) -> tuple[np.ndarray, str | None]:
        if self.mean is None or self.std is None:
            raise ValueError("RecordingMeanTrimmedZScore must be fitted during recording preparation.")
        return (np.asarray(values, dtype=float) - self.mean) / self.std, "dimensionless"
