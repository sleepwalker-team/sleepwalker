import numpy as np
import pandas as pd

class ChannelDropout:
    """Randomly drop entire channels (columns) with probability ``p``.

    Each column is independently kept with probability ``1 - p``. The returned
    frame may have fewer columns than the input; at least the surviving columns
    are returned in their original order.

    Args:
        p: Probability that a given channel is dropped.
    """

    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        keep_mask = np.random.rand(len(X.columns)) >= self.p
        kept_cols = X.columns[keep_mask]
        return X[kept_cols]