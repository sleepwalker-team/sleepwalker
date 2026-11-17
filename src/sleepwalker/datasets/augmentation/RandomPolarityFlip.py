import numpy as np
import pandas as pd

class RandomPolarityFlip:
    """Randomly flip the sign of individual channels.

    The augmentation is applied to a sample with probability ``p``; when
    applied, each channel is independently negated with probability
    ``flip_p``.

    Args:
        p: Probability that the augmentation is applied to the sample at all.
        flip_p: Probability that each individual channel is flipped.
    """

    def __init__(self, p=0.5, flip_p=0.1):
        self.p = p
        self.flip_p = flip_p

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        if np.random.rand() > self.p:
            return X  # no-op

        X_aug = X.copy()

        for col in X.columns:
            if np.random.rand() < self.flip_p:
                X_aug[col] = -X_aug[col]

        return X_aug