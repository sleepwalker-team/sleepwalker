import numpy as np
import pandas as pd
import random
from typing import Literal

class TimeShiftAndCrop:
    """Randomly time-shift a signal and crop a fixed-duration segment.

    The frame is shifted by a random whole-sample offset up to ``max_shift``
    (edges filled with the nearest value), then a segment of duration
    ``output_size`` is cropped at ``crop_pos``.

    Args:
        output_size: Duration of the cropped segment, as a string accepted by
            ``pandas.to_timedelta`` (e.g. ``"30s"``).
        sampling_frequency: Sampling rate in Hz used to convert durations to
            sample counts.
        max_shift: Maximum absolute shift duration, as a string accepted by
            ``pandas.to_timedelta``.
        fill: Edge fill strategy. Only ``"closest"`` is supported; any other
            value raises ``ValueError``.
        crop_pos: Where to crop the segment: ``"left"``, ``"middle"``, or
            ``"right"``.

    Raises:
        ValueError: If ``fill`` is not ``"closest"``, or if the requested crop
            length exceeds the input length.
    """

    def __init__(
        self,
        output_size,
        sampling_frequency,
        max_shift,
        fill='closest',
        crop_pos: Literal["left", "middle", "right"] = "middle",
    ):
        self.output_size = pd.to_timedelta(output_size)
        self.sampling_frequency = sampling_frequency
        self.max_shift = pd.to_timedelta(max_shift)
        self.fill = fill
        self._max_shift_samples = int(self.max_shift.total_seconds() * self.sampling_frequency)
        self.crop_pos = crop_pos
        self.len = int(self.output_size.total_seconds() * self.sampling_frequency)

        if fill != "closest":
            raise ValueError(f"Unknown fill received. Received {fill}, but support only {{closest}}")

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        X = data.copy()
        T, D = X.shape
        shift = random.randint(-self._max_shift_samples, self._max_shift_samples)

        if shift != 0:
            shifted = np.empty_like(X.values)

            if shift > 0:
                # shift right
                shifted[shift:, :] = X.values[:-shift, :]
                shifted[:shift, :] = X.values[0:1, :].repeat(shift, axis=0)
            else:
                # shift left
                shifted[:shift, :] = X.values[-shift:, :]
                shifted[shift:, :] = X.values[-1:, :].repeat(-shift, axis=0)

            X = pd.DataFrame(shifted, index=X.index, columns=X.columns)

        # --- Crop ---
        L = self.len
        if L > T:
            raise ValueError(f"Requested crop length {L} exceeds input length {T}.")

        if self.crop_pos == "middle":
            start = (T - L) // 2
        elif self.crop_pos == "left":
            start = 0
        elif self.crop_pos == "right":
            start = T - L
        else:
            raise ValueError(f"Invalid crop location: {self.crop_pos}")

        end = start + L
        cropped = X.iloc[start:end].copy()

        return cropped