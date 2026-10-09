"""Callable channel processors with optional recording fitting."""

from abc import ABC, abstractmethod

import numpy as np


class Normalizer(ABC):
    """Process one (N, 1) channel, with its current unit and input scope.

    Application returns (values, unit), or None to reject this recording or
    window alias. Optional fit methods update state in place and return self
    for success, or None for recording rejection. The framework calls fit
    only with is_recording=True, independently for each recording and alias.
    """

    @abstractmethod
    def __call__(self, values: np.ndarray, *, unit: str | None, is_recording: bool) -> tuple[np.ndarray, str | None] | None:
        ...
