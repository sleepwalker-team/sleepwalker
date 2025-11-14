import numpy as np
import pandas as pd

class GaussianNoise:
    """
    Adds Gaussian noise directly in the time domain.
    """
    def __init__(self, mu=0, sigma=0.1, **kwargs):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.to_numpy(dtype=np.float32)
        noise = self.mu + np.sqrt(self.sigma) * np.random.randn(*data.shape)
        out = data + noise
        return pd.DataFrame(out, index=X.index, columns=X.columns)