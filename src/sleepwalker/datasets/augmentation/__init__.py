"""Random signal augmentations applied to a sample during training.

Each augmentation is a plain callable that takes a :class:`pandas.DataFrame` of
window samples (rows are time steps, columns are channels) and returns a
transformed ``DataFrame``. All preserve the input shape except
:class:`ChannelDropout`, which may drop columns. Augmentations are intended for
the training split only and must not be applied at inference time.
"""

from .AmplitudeScale import AmplitudeScale
from .ChannelDropout import ChannelDropout
from .ChannelShift import ChannelShift
from .FrequencyNoise import FrequencyNoise
from .GaussianNoise import GaussianNoise
from .RandomPolarityFlip import RandomPolarityFlip
from .RandomResampleJitter import RandomResampleJitter
from .TimeShiftAndCrop import TimeShiftAndCrop

__all__ = [
    "AmplitudeScale",
    "ChannelDropout",
    "ChannelShift",
    "FrequencyNoise",
    "GaussianNoise",
    "RandomPolarityFlip",
    "RandomResampleJitter",
    "TimeShiftAndCrop",
]
