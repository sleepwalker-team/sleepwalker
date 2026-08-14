import numpy as np

from abc import ABC, abstractmethod

class Normalizer(ABC):
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...
