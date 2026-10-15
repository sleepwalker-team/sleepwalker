"""Numerical building blocks for the DRIVEN respiratory window baseline."""

import cloudpickle
import numpy as np
import pytest
import torch

from sleepwalker.datasets.normalizer import RecordingMeanTrimmedZScore
from sleepwalker.models.preprocessors.DelayedChannelSlices import DelayedChannelSlices
from sleepwalker.models.preprocessors.ScaleChannels import ScaleChannels


def test_trimmed_recording_statistics_and_window_application():
    values = np.r_[np.arange(-9, 10), 100.0].reshape(-1, 1)
    scaler = RecordingMeanTrimmedZScore()
    assert scaler.fit(values, unit="arbitrary", is_recording=True) is scaler
    assert scaler.mean == pytest.approx(0.0)
    assert scaler.std == pytest.approx(np.std(np.arange(-9, 10), ddof=1))
    window = np.array([[3.0], [6.0]])
    scaled, unit = scaler(window, unit="arbitrary", is_recording=False)
    np.testing.assert_allclose(scaled, window / scaler.std)
    assert scaled.shape == window.shape and unit == "dimensionless"
    restored = cloudpickle.loads(cloudpickle.dumps(scaler))
    np.testing.assert_allclose(restored(window, unit=None, is_recording=False)[0], scaled)


def test_trimmed_rounding_and_cutoff_ties_use_recording_order():
    # Full mean is zero; retain floor(5 * .7) = 3 of the equally distant samples.
    scaler = RecordingMeanTrimmedZScore(retained_fraction=0.7)
    assert scaler.fit(np.array([-1., 1., -1., 1., 0.]), unit=None, is_recording=True) is scaler
    expected = np.array([-1., 1., 0.])
    assert scaler.mean == pytest.approx(expected.mean())
    assert scaler.std == pytest.approx(expected.std(ddof=1))


@pytest.mark.parametrize("values", [[], [1], [1, 2], [1, 1, 1], [1, 2, np.nan], [1, 2, np.inf]])
def test_trimmed_unusable_recording_rejection_clears_previous_fit(values):
    scaler = RecordingMeanTrimmedZScore()
    scaler.fit(np.arange(20), unit=None, is_recording=True)
    assert scaler.fit(np.array(values), unit=None, is_recording=True) is None
    with pytest.raises(ValueError, match="must be fitted"):
        scaler(np.array([1.0]), unit=None, is_recording=False)


@pytest.mark.parametrize("fraction", [0, -1, 1.1, np.nan, np.inf])
def test_trimmed_invalid_retention(fraction):
    with pytest.raises(ValueError, match="retained_fraction"):
        RecordingMeanTrimmedZScore(fraction)


def test_trimmed_full_retention_uses_sample_standard_deviation():
    scaler = RecordingMeanTrimmedZScore(1)
    scaler.fit(np.array([1., 3.]), unit=None, is_recording=True)
    assert scaler.mean == 2
    assert scaler.std == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError, match="complete recording"):
        scaler.fit(np.array([1., 3.]), unit=None, is_recording=False)
    with pytest.raises(ValueError, match="one channel"):
        scaler.fit(np.zeros((3, 2)), unit=None, is_recording=True)


def test_driven_reference_geometry_delays_and_scaling():
    preprocessor = torch.nn.Sequential(
        ScaleChannels(scales={1: 0.01}, clip={1: [0, 1]}),
        DelayedChannelSlices(sampling_frequency=64, window_duration=30, slices=[(0, 0), (1, 0), (1, 10), (1, 15), (1, 20), (1, 25)]),
    )
    time = torch.arange(3520, dtype=torch.float64)
    data = torch.stack([time, time / 64], dim=-1).unsqueeze(0).repeat(2, 1, 1)
    original = data.clone()
    result = preprocessor(data)
    assert result.shape == (2, 1920, 6)
    assert result.dtype == data.dtype and result.device == data.device
    torch.testing.assert_close(result[:, :, 0], data[:, :1920, 0])
    for branch, seconds in enumerate([0, 10, 15, 20, 25], start=1):
        torch.testing.assert_close(result[:, :, branch], data[:, seconds * 64:seconds * 64 + 1920, 1] / 100)
    torch.testing.assert_close(data, original)
    for step in preprocessor:
        assert step.requires_warmup() is False
        step.update(data)
    restored = cloudpickle.loads(cloudpickle.dumps(preprocessor))
    torch.testing.assert_close(restored(data), result)


def test_driven_configurable_geometry_delay_order_clipping_and_extra_context():
    preprocessor = torch.nn.Sequential(
        ScaleChannels(scales={1: 0.01}, clip={1: [0, 1]}),
        DelayedChannelSlices(sampling_frequency=2, window_duration=2, slices=[(0, 0), (1, 1), (1, 0)]),
    )
    data = torch.tensor([[[1., -10.], [2., 90.], [3., 100.], [4., 120.], [5., 80.], [6., 50.], [7., 20.]]])
    expected = torch.tensor([[[1., 1., 0.], [2., 1., .9], [3., .8, 1.], [4., .5, 1.]]])
    torch.testing.assert_close(preprocessor(data), expected)
    with pytest.raises(ValueError, match="at least 6 context samples"):
        preprocessor(data[:, :5])


