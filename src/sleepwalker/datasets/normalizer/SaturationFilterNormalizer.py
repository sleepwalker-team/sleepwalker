from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer


class SaturationFilterNormalizer(SignalFilterNormalizer):
    """Low-pass normalizer with clipping tuned for blood-oxygen saturation.

    Configures `SignalFilterNormalizer` as a 0.4 Hz low-pass (no lower cutoff)
    with no notch filter, and clips the raw signal to the 50.0-100.0 percent
    range before filtering, suitable for SpO2 channels.

    Args:
        fs: Sampling rate of the saturation signal in Hz.
        highcut: Low-pass cutoff frequency in Hz (default 0.4).
        band_order: Order of the Butterworth low-pass filter (default 2).
        clip_range: ``(min, max)`` tuple the raw signal is clipped to before
            filtering (default ``(50.0, 100.0)``, i.e. percent saturation).
        **kwargs: Additional keyword arguments forwarded to
            `SignalFilterNormalizer` (e.g. `notch_freq`, `normalize`, `mean`,
            `std`).
    """

    def __init__(self, fs, highcut=0.4, band_order=2, clip_range=(50.0, 100.0), **kwargs):
        super().__init__(fs=fs, highcut=highcut, band_order=band_order, clip_range=clip_range, **kwargs)

    def __call__(self, values, *, unit: str | None, is_recording: bool):
        # Clipping is meaningful only for physical percentage values. Counts
        # or an unknown waveform scale need an explicit correction first.
        if unit not in {"%", "percent"}:
            return None
        return super().__call__(values, unit=unit, is_recording=is_recording)
