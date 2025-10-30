from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod

class Normalizer(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray) -> Normalizer:
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...