"""The one model-facing featurizer: builds extractors by name and runs them behind one shared config.

    featurizer = Featurizer(["minirocket"], channel_roles, 100)

All extractors share one immutable ``ExtractionContext``, so they agree on the window by construction.
``fit`` fans out to every extractor, and ``transform_batch`` concatenates their outputs.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from sleepwalker.features.context import ExtractionContext
from sleepwalker.features.extractors import get_extractor


class Featurizer:
    """Builds and composes extractors by name behind one shared config.

    Args:
        extractors: Registered extractor names, e.g. ``["minirocket"]`` (see the ``extractors/`` folder).
        channel_roles: The ``[B, T, C]`` batch's ``C`` axis (``dataset.get_input_channels()``).
        sample_frequency: Sampling rate of the window in Hz.
        epoch_duration: Length of the scored epoch, taken from the centre of the window.
        options: Per-extractor keyword args keyed by name, e.g. ``{"multirocket": {"num_kernels": 10000}}``.
    """

    def __init__(
        self,
        extractors: Sequence[str],
        channel_roles: Sequence[str],
        sample_frequency: float,
        *,
        epoch_duration: str = "30s",
        options: Optional[Mapping[str, Mapping[str, object]]] = None,
    ) -> None:
        names = list(extractors)
        if not names:
            raise ValueError("Featurizer needs at least one extractor.")
        channel_roles = list(channel_roles)
        if len(set(channel_roles)) != len(channel_roles):
            raise ValueError(f"channel_roles must be unique, got {channel_roles}.")

        self.context = ExtractionContext(
            channel_roles=channel_roles,
            sample_frequency=float(sample_frequency),
            epoch_duration=pd.to_timedelta(epoch_duration),
        )

        options = options or {}
        self.extractors = [
            get_extractor(name, self.context, **dict(options.get(name, {}))) for name in names
        ]

        feature_names = [n for extractor in self.extractors for n in extractor.feature_names]
        if len(set(feature_names)) != len(feature_names):
            dupes = sorted({n for n in feature_names if feature_names.count(n) > 1})
            raise ValueError(f"Extractors produced duplicate feature names: {dupes}.")
        self._feature_names = feature_names

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    @property
    def fitted(self) -> bool:
        """Whether every extractor can transform -- the single source of truth for fitted state."""
        return all(extractor.fitted for extractor in self.extractors)

    # ---- the featurizer surface ---- #
    def fit(self, x: np.ndarray, y=None) -> "Featurizer":
        """Fit every extractor on a ``[B, T, C]`` batch (no-op for stateless ones), then return self."""
        for extractor in self.extractors:
            extractor.fit(x, y)
        return self

    def transform_window(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([e.transform_window(x) for e in self.extractors], axis=0)

    def transform_batch(self, x: np.ndarray, n_jobs: int = 1) -> np.ndarray:
        """Featurize a ``[B, T, C]`` batch into ``[B, n_features]`` on ``n_jobs`` CPUs."""
        if x.ndim != 3:
            raise ValueError(f"Expected a [B, T, C] batch, got shape {x.shape}.")
        if len(x) == 0:
            return np.empty((0, self.n_features), dtype=float)
        return np.concatenate([e.transform_batch(x, n_jobs=n_jobs) for e in self.extractors], axis=1)
