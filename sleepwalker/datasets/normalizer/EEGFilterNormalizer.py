import numpy as np

from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from sleepwalker.datasets.normalizer import Normalizer

class EEGFilterNormalizer(Normalizer):
    def __init__(self, fs, lowcut=0.3, highcut=35.0, notch_freq=50.0, band_order=4, notch_q=30, **kwargs):
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.band_order = band_order
        self.notch_q = notch_q
        self.fs = fs
        self.nyq = 0.5 * fs
        self.sos_band = butter(self.band_order, [self.lowcut / self.nyq, self.highcut / self.nyq],
                       btype="band", output="sos")
        # self.b_band, self.a_band = butter(self.band_order,[self.lowcut / self.nyq, self.highcut / self.nyq], btype='band') 
        if self.notch_freq:
            self.b_notch, self.a_notch = iirnotch(self.notch_freq / self.nyq, self.notch_q)

    def _filter(self, signal):
        # filtered = filtfilt(self.b_band, self.a_band, signal)
        filtered = sosfiltfilt(self.sos_band, signal)

        if self.notch_freq:
            filtered = filtfilt(self.b_notch, self.a_notch, filtered)

        return filtered

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError("Expected shape (N, 1), got {}".format(X.shape))

        signal = X[:, 0]
        filtered = self._filter(signal)

        self.mean_ = np.mean(filtered)
        self.std_ = np.std(filtered) if np.std(filtered) > 1e-7 else 1.0
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError("Expected shape (N, 1), got {}".format(X.shape))

        signal = X[:, 0]
        filtered = self._filter(signal)

        normalized = (filtered - self.mean_) / self.std_
        return normalized.reshape(-1, 1)