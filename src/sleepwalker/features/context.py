"""Shared, immutable extraction config, built once by the ``Featurizer`` and handed to every extractor.

Because all extractors read the same context, they agree on the window by construction. Extractor-specific
options live in the featurizer's ``options={name: {...}}``, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExtractionContext:
    """Immutable shared config handed to every extractor.

    Args:
        channel_roles: Names of the input channels, in the order of the ``[B, T, C]`` batch's ``C`` axis.
        sample_frequency: Sampling rate of the window in Hz.
        epoch_duration: Length of the scored epoch, taken from the centre of the window.
    """

    channel_roles: list[str]
    sample_frequency: float
    epoch_duration: pd.Timedelta

    @property
    def window_samples(self) -> int:
        """Number of time samples in the scored epoch (``epoch_duration × sample_frequency``)."""
        return int(round(self.epoch_duration.total_seconds() * self.sample_frequency))

