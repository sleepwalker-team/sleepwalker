from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer


class EEGFilterNormalizer(SignalFilterNormalizer):
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
