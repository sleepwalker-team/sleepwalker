import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from sleepwalker.datasets.normalizer import Normalizer


class SignalFilterNormalizer(Normalizer):
    """Zero-phase Butterworth band/high/low-pass filter with optional notch and standardization.

    Builds its filters once at construction time and applies them to a single
    signal channel on each call to `transform`. The filtering chain is:
    optional clipping, then the Butterworth band/high/low-pass filter, then an
    optional IIR notch filter. All filtering is zero-phase (forward-backward).
    When `normalize` is set, the filtered signal is additionally standardized
    as ``(x - mean) / std``.

    The Butterworth response depends on which cutoffs are provided:
    a band-pass when both `lowcut` and `highcut` are given, a high-pass when
    only `lowcut` is given, and a low-pass when only `highcut` is given. If
    neither is set, no band filter is applied. The notch filter is applied only
    when `notch_freq` is truthy.

    Args:
        fs: Sampling rate of the signal in Hz. Cutoff and notch frequencies are
            normalized against the Nyquist frequency ``0.5 * fs``.
        lowcut: Lower cutoff frequency in Hz, or ``None`` for no lower bound.
        highcut: Upper cutoff frequency in Hz, or ``None`` for no upper bound.
        notch_freq: Center frequency of the IIR notch filter in Hz. A falsy
            value (``None`` or ``0``) disables the notch.
        band_order: Order of the Butterworth band/high/low-pass filter.
        notch_q: Quality factor of the IIR notch filter; higher is narrower.
        clip_range: Optional ``(min, max)`` tuple to clip the raw signal to
            before filtering, or ``None`` to skip clipping.
        normalize: Whether to standardize the filtered signal with `mean` and
            `std` after filtering.
        mean: Mean used for standardization. Must be finite.
        std: Standard deviation used for standardization. Must be finite and
            strictly positive.

    Raises:
        ValueError: If `mean` is not finite, or `std` is not finite and positive.
    """

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
        """Apply clipping, the band filter, and the notch filter to a 1D signal.

        Args:
            signal: Input signal, coerced to a float array.

        Returns:
            The filtered signal as a float array with the same length as the
            input, in the same physical units as the input.
        """
        filtered = np.asarray(signal, dtype=float)

        if self.clip_range is not None:
            filtered = np.clip(filtered, self.clip_range[0], self.clip_range[1])

        if self.sos_band is not None:
            filtered = sosfiltfilt(self.sos_band, filtered)

        if self.notch_freq:
            filtered = filtfilt(self.b_notch, self.a_notch, filtered)

        return filtered

    def transform(self, X):
        """Filter and optionally standardize a single-channel signal.

        Args:
            X: Input array of shape ``(N, 1)``, where the single column holds
                the signal samples.

        Returns:
            Array of shape ``(N, 1)`` containing the filtered signal, scaled by
            ``(x - mean) / std`` when `normalize` is set.

        Raises:
            ValueError: If `X` is not 2D with a single column.
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError(f"Expected shape (N, 1), got {X.shape}")

        filtered = self.filter(X[:, 0])
        if self.normalize:
            filtered = (filtered - self.mean) / self.std
        return filtered.reshape(-1, 1)
