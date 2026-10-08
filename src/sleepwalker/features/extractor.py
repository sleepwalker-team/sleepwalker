"""The extractor contract: implement ``feature_names`` and ``transform_window`` (one ``[T, C]`` window in,
one ``[n_features]`` vector out), and batching with joblib parallelism comes for free.

Shared config (channel roles, sample frequency, ...) is read from ``self.context``. Stateful extractors
override ``fit``. Register a new one with ``@register_extractor(name)`` in ``extractors/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed

from sleepwalker.features.context import ExtractionContext


class Extractor(ABC):
    """Fixed-width featurization of a signal window, configured by a shared context.

    Args:
        context: The shared :class:`ExtractionContext` (channel_roles, sample_frequency, ...).
        **options: Extractor-specific keyword args (e.g. ``num_kernels`` for Rocket). Subclasses
            declare the ones they accept; the base ignores extras.
    """

    def __init__(self, context: ExtractionContext, **options) -> None:
        self.context = context

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Ordered feature column names; length equals ``n_features``."""

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def fitted(self) -> bool:
        """Whether this extractor can transform. Stateless extractors always can; stateful ones override."""
        return True

    def fit(self, x: np.ndarray, y=None) -> "Extractor":
        """Fit any state from a ``[B, T, C]`` batch. No-op by default; stateful extractors override."""
        return self

    @abstractmethod
    def transform_window(self, x: np.ndarray) -> np.ndarray:
        """Featurize one ``[T, C]`` window into a ``[n_features]`` vector."""

    def transform_batch(self, x: np.ndarray, n_jobs: int = 1) -> np.ndarray:
        """Featurize a ``[B, T, C]`` batch into ``[B, n_features]`` by looping over windows.

        Serial when ``n_jobs == 1``; otherwise spreads windows across ``n_jobs`` processes with joblib
        (loky). Batch-native extractors (e.g. MultiRocket) override this.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected a [B, T, C] batch, got shape {x.shape}.")
        if len(x) == 0:
            return np.empty((0, self.n_features), dtype=float)

        if n_jobs == 1:
            rows = [self.transform_window(window) for window in x]
        else:
            rows = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self.transform_window)(window) for window in x
            )
        return np.stack(rows, axis=0)
