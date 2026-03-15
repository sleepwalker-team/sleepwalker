import numpy as np
import pandas as pd

class RandomResampleJitter:
    def __init__(self, p=0.5, scale=0.1):
        """
        p: probability to apply the augmentation
        scale: maximum relative resampling factor.
               alpha is sampled from [1-scale, 1+scale]
        """
        self.p = p
        self.scale = scale

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        if np.random.rand() > self.p:
            return X  # no-op

        # original timeline
        orig_index = X.index
        t = np.linspace(0, 1, len(X))

        # sample resampling factor α
        alpha = np.random.uniform(1 - self.scale, 1 + self.scale)
        new_t = np.linspace(0, 1, int(len(X) * alpha))

        # resample each column
        X_new = np.vstack([
            np.interp(new_t, t, X[col].values)
            for col in X.columns
        ]).T

        # convert back to original number of samples
        # by interpolating to original timestamps
        X_back = np.vstack([
            np.interp(t, new_t, X_new[:, i])
            for i in range(X_new.shape[1])
        ]).T

        return pd.DataFrame(X_back, index=orig_index, columns=X.columns)
