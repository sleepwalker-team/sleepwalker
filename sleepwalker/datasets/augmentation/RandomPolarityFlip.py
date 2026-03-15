import numpy as np
import pandas as pd

class RandomPolarityFlip:
    def __init__(self, p=0.5, flip_p=0.1):
        """
        p: probability to apply augmentation to the sample
        flip_p: probability that each channel is flipped
        """
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