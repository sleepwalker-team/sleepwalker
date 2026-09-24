import numpy as np
import pandas as pd

class AmplitudeScale:
    """Scale the whole sample by a random factor drawn uniformly per call.

    Multiplies every value in the frame by a single scalar sampled from
    ``[min, max]``. The same factor is applied to all channels and time steps
    within one call.

    Args:
        min: Lower bound of the sampled scale factor.
        max: Upper bound of the sampled scale factor.
    """

    def __init__(self, min=1.0, max=1.1, **kwargs):
        self.min = min
        self.max = max

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        # TODO Sample a per-channel scale?
        data = X.to_numpy(dtype=np.float32)
        out = data * np.random.uniform(low=self.min, high=self.max)
        return pd.DataFrame(out, index=X.index, columns=X.columns)