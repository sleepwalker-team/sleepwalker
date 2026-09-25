import numpy as np
import pandas as pd

class FrequencyNoise:
    """Add Gaussian noise to the signal in the frequency domain.

    Transforms the frame with a real FFT, adds independent Gaussian noise to
    both the real and imaginary components, and transforms back to the time
    domain. This perturbs the spectral content rather than the raw samples.

    Args:
        mean: Mean of the Gaussian noise added to each spectral component.
        std: Standard deviation of the Gaussian noise added to each spectral
            component.
    """

    def __init__(self, mean=0.0, std=0.1, **kwargs):
        self.mean = mean
        self.std = std

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.to_numpy(dtype=np.float32)
        T, d = data.shape

        freq_domain = np.fft.rfft(data, axis=0)
        noise_real = np.random.randn(*freq_domain.real.shape) * self.std + self.mean
        noise_imag = np.random.randn(*freq_domain.imag.shape) * self.std + self.mean
        freq_domain.real += noise_real
        freq_domain.imag += noise_imag

        time_domain = np.fft.irfft(freq_domain, n=T, axis=0)
        return pd.DataFrame(time_domain, index=X.index, columns=X.columns)