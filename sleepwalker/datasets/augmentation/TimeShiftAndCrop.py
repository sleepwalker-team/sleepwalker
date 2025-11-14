import numpy as np
import pandas as pd
import random
from typing import Literal

class TimeShiftAndCrop:
    """
    Randomly time-shifts a signal (DataFrame) by up to `max_shift` 
    and crops a segment of duration `output_size`.
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