import numpy as np

from abc import ABC, abstractmethod

class Normalizer(ABC):
    """Abstract base class for signal normalizers.

    A normalizer maps a raw input array to a processed output array via the
    `transform` method. Concrete subclasses implement the actual filtering
    and scaling; see `SignalFilterNormalizer` for the filtering pipeline used
    across the bundled normalizers.
    """

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform a raw input array into its normalized form.

        Args:
            X: Input signal array.

        Returns:
            The transformed signal array.
        """
        ...
