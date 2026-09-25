from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer


class PulseFilterNormalizer(SignalFilterNormalizer):
    """Band-pass normalizer tuned for pulse / plethysmography signals.

    Configures `SignalFilterNormalizer` with a 0.5-8.0 Hz band-pass and no
    notch filter, suitable for pulse-rate channels.

    Args:
        fs: Sampling rate of the pulse signal in Hz.
        lowcut: Lower band-pass cutoff in Hz (default 0.5).
        highcut: Upper band-pass cutoff in Hz (default 8.0).
        band_order: Order of the Butterworth band-pass filter (default 4).
        **kwargs: Additional keyword arguments forwarded to
            `SignalFilterNormalizer` (e.g. `notch_freq`, `clip_range`,
            `normalize`, `mean`, `std`).
    """

    def __init__(self, fs, lowcut=0.5, highcut=8.0, band_order=4, **kwargs):
        super().__init__(fs=fs, lowcut=lowcut, highcut=highcut, band_order=band_order, **kwargs)
