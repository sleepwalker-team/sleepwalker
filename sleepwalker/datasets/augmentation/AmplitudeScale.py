import numpy as np
import pandas as pd

class AmplitudeScale:
    def __init__(self, min=1.0, max=1.1, **kwargs):
        self.min = min
        self.max = max

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        # TODO Sample a per-channel scale?
        data = X.to_numpy(dtype=np.float32)
        out = data * np.random.uniform(low=self.min, high=self.max)
        return pd.DataFrame(out, index=X.index, columns=X.columns)