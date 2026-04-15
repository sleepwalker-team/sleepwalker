from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer


class RespirationFilterNormalizer(SignalFilterNormalizer):
    def __init__(self, fs, lowcut=0.05, highcut=3.0, band_order=4, **kwargs):
        super().__init__(fs=fs, lowcut=lowcut, highcut=highcut, band_order=band_order)
