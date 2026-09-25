from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer


class EEGFilterNormalizer(SignalFilterNormalizer):
    """Band-pass plus line-noise notch normalizer tuned for EEG signals.

    Configures `SignalFilterNormalizer` with a 0.3-35.0 Hz band-pass and a
    50.0 Hz notch (mains hum), suitable for electroencephalography channels.

    Args:
        fs: Sampling rate of the EEG signal in Hz.
        lowcut: Lower band-pass cutoff in Hz (default 0.3).
        highcut: Upper band-pass cutoff in Hz (default 35.0).
        notch_freq: Notch center frequency in Hz (default 50.0).
        band_order: Order of the Butterworth band-pass filter (default 4).
        notch_q: Quality factor of the notch filter (default 30).
        **kwargs: Additional keyword arguments forwarded to
            `SignalFilterNormalizer` (e.g. `clip_range`, `normalize`, `mean`,
            `std`).
    """

    def __init__(self, fs, lowcut=0.3, highcut=35.0, notch_freq=50.0, band_order=4, notch_q=30, **kwargs):
        super().__init__(
            fs=fs,
            lowcut=lowcut,
            highcut=highcut,
            notch_freq=notch_freq,
            band_order=band_order,
            notch_q=notch_q,
            **kwargs,
        )
