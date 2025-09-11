from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod

class Normalizer(ABC):
    """
    Abstract base class for per-channel normalization that may depend on
    the EDF (source) sampling frequency and the (target) resample frequency.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, fs: float) -> "Normalizer":
        """
        Fit on a (N, 1) array at **EDF native** sampling rate.
        """
        ...

    @abstractmethod
    def transform(self, X: np.ndarray, fs: float) -> np.ndarray:
        """
        Transform a (N, 1) array. Unless documented otherwise, X is assumed to be
        at **EDF native** sampling rate (before resampling).
        """
        ...
