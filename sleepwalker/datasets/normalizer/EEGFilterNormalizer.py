import numpy as np

from scipy.signal import butter, filtfilt, iirnotch

from sleepwalker.datasets.normalizer import Normalizer

class EEGFilterNormalizer(Normalizer):
    def __init__(self, lowcut=0.3, highcut=35.0, notch_freq=50.0, band_order=4, notch_q=30, **kwargs):
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.band_order = band_order
        self.notch_q = notch_q

    def _filter(self, signal, fs):
        nyq = 0.5 * fs
        b_band, a_band = butter(self.band_order,[self.lowcut / nyq, self.highcut / nyq], btype='band') # type: ignore
        filtered = filtfilt(b_band, a_band, signal)

        if self.notch_freq:
            b_notch, a_notch = iirnotch(self.notch_freq / nyq, self.notch_q)
            filtered = filtfilt(b_notch, a_notch, filtered)

        return filtered

    def fit(self, X, fs):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError("Expected shape (N, 1), got {}".format(X.shape))

        signal = X[:, 0]
        filtered = self._filter(signal, fs)

        self.mean_ = np.mean(filtered)
        self.std_ = np.std(filtered) if np.std(filtered) > 0 else 1.0
        return self

    def transform(self, X, fs):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError("Expected shape (N, 1), got {}".format(X.shape))

        signal = X[:, 0]
        filtered = self._filter(signal, fs)

        normalized = (filtered - self.mean_) / self.std_
        return normalized.reshape(-1, 1)