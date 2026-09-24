import pandas as pd
import numpy as np

class ChannelShift:
    """Randomly shift individual channels (columns) left or right in time.

    Each channel is shifted with probability ``p_shift`` by a length drawn from
    a Poisson distribution with mean ``p_shift_len`` (capped at half the frame
    length). The vacated samples are filled according to ``fill``.

    Args:
        p_left: Probability of shifting left rather than right, in ``(0, 1)``.
        fill: How to fill the vacated samples: ``"closest"`` repeats the nearest
            edge value, ``"zeros"`` fills with ``0.0``.
        p_shift_len: Mean of the Poisson distribution from which the shift
            length (in samples) is drawn. Must be positive.
        p_shift: Probability that a given channel is shifted at all, in
            ``(0, 1)``.
    """

    def __init__(self, p_left=0.5, fill='closest', p_shift_len=5, p_shift=0.5, **kwargs):
        assert 0 < p_left < 1, "p_left must be in (0,1)"
        assert 0 < p_shift < 1, "p_shift must be in (0,1)"
        assert p_shift_len > 0, "p_shift_len must be > 0"
        self.p_left = p_left
        self.fill = fill
        self.p_shift = p_shift
        self.p_shift_len = p_shift_len

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        max_shift = len(X) // 2

        for c in X.columns:
            if np.random.rand() < self.p_shift:
                shift_amount = np.random.poisson(self.p_shift_len)
                shift_amount = min(shift_amount, max_shift)
                if shift_amount == 0:
                    continue

                direction = np.random.choice(['left', 'right'], p=[self.p_left, 1 - self.p_left])
                col = X[c].values

                if direction == 'left':
                    shifted = np.empty_like(col)
                    shifted[:-shift_amount] = col[shift_amount:]
                    if self.fill == 'zeros':
                        shifted[-shift_amount:] = 0.0
                    elif self.fill == 'closest':
                        shifted[-shift_amount:] = col[-1]
                else:  # right
                    shifted = np.empty_like(col)
                    shifted[shift_amount:] = col[:-shift_amount]
                    if self.fill == 'zeros':
                        shifted[:shift_amount] = 0.0
                    elif self.fill == 'closest':
                        shifted[:shift_amount] = col[0]

                X_out[c] = shifted

        return X_out
