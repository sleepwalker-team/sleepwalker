"""Explicit conversions between known, compatible channel units."""

from sleepwalker.core.signal import unit_conversion_factor
from sleepwalker.datasets.normalizer.base import Normalizer


class ConvertUnit(Normalizer):
    """Convert known source units; reject unknown or incompatible calibration.

    An explicit preceding processor must correct documented source labels
    first. This step cannot infer a gain for counts or relative waveforms.
    An invalid target fails at construction instead of excluding patient data.
    """

    def __init__(self, target):
        unit_conversion_factor(target, target, assume_if_missing=False)
        self.target = target

    def __call__(self, values, *, unit: str | None, is_recording: bool):
        try:
            factor = unit_conversion_factor(unit, self.target, assume_if_missing=False)
        except ValueError:
            return None
        return values * factor, self.target
