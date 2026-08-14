import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from sleepwalker.datasets.normalizer import Normalizer


class SignalFilterNormalizer(Normalizer):
    def __init__(
        self,
        fs,
        lowcut=None,
        highcut=None,
        notch_freq=None,
        band_order=4,
        notch_q=30,
        clip_range=None,
        normalize=True,
        mean=0.0,
        std=1.0,
    ):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.band_order = band_order
        self.notch_q = notch_q
        self.clip_range = clip_range
        self.normalize = bool(normalize)
        self.mean = float(mean)
        self.std = float(std)
        if not np.isfinite(self.mean):
            raise ValueError("mean must be finite.")
        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("std must be finite and positive.")
        self.nyq = 0.5 * fs

        self.sos_band = None
        if self.lowcut is not None and self.highcut is not None:
            self.sos_band = butter(
                self.band_order,
                [self.lowcut / self.nyq, self.highcut / self.nyq],
                btype="band",
                output="sos",
            )
        elif self.lowcut is not None:
            self.sos_band = butter(self.band_order, self.lowcut / self.nyq, btype="high", output="sos")
        elif self.highcut is not None:
            self.sos_band = butter(self.band_order, self.highcut / self.nyq, btype="low", output="sos")

        if self.notch_freq:
            self.b_notch, self.a_notch = iirnotch(self.notch_freq / self.nyq, self.notch_q)

    def filter(self, signal):
        filtered = np.asarray(signal, dtype=float)

        if self.clip_range is not None:
            filtered = np.clip(filtered, self.clip_range[0], self.clip_range[1])

        if self.sos_band is not None:
            filtered = sosfiltfilt(self.sos_band, filtered)

        if self.notch_freq:
            filtered = filtfilt(self.b_notch, self.a_notch, filtered)

        return filtered

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError(f"Expected shape (N, 1), got {X.shape}")

        filtered = self.filter(X[:, 0])
        if self.normalize:
            filtered = (filtered - self.mean) / self.std
        return filtered.reshape(-1, 1)
