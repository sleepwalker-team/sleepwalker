import numpy as np
import pandas as pd

class ChannelDropout:
    """
    Randomly drops entire channels (columns) with probability p.
    """
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        keep_mask = np.random.rand(len(X.columns)) >= self.p
        kept_cols = X.columns[keep_mask]
        return X[kept_cols]