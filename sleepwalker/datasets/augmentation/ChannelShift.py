import pandas as pd
import numpy as np

class ChannelShift:
    """
    Randomly shifts entire channels (columns) left or right.
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
