import numpy as np
import pandas as pd

class GaussianNoise:
    """Add Gaussian noise directly in the time domain.

    Adds ``mu + sqrt(sigma) * N(0, 1)`` to every sample. Note that ``sigma`` is
    a variance-like parameter: the standard deviation of the added noise is
    ``sqrt(sigma)``.

    Args:
        mu: Mean of the added noise.
        sigma: Variance-like scale of the added noise; the noise standard
            deviation is ``sqrt(sigma)``.
    """

    def __init__(self, mu=0, sigma=0.1, **kwargs):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.to_numpy(dtype=np.float32)
        noise = self.mu + np.sqrt(self.sigma) * np.random.randn(*data.shape)
        out = data + noise
        return pd.DataFrame(out, index=X.index, columns=X.columns)