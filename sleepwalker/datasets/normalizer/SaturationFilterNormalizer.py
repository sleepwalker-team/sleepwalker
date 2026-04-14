from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer


class SaturationFilterNormalizer(SignalFilterNormalizer):
    def __init__(self, fs, highcut=0.4, band_order=2, clip_range=(50.0, 100.0), **kwargs):
        super().__init__(fs=fs, highcut=highcut, band_order=band_order, clip_range=clip_range)
