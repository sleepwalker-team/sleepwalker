"""Configurable channel geometry, scaling and composition independent of DRIVEN."""

import cloudpickle
import pytest
import torch

from sleepwalker.config import build_value
from sleepwalker.models.preprocessors.DelayedChannelSlices import DelayedChannelSlices
from sleepwalker.models.preprocessors.ScaleChannels import ScaleChannels


def test_arbitrary_channels_order_and_repeated_slices_without_implicit_scaling():
    data = torch.arange(2 * 7 * 4, dtype=torch.float64).reshape(2, 7, 4)
    step = DelayedChannelSlices(sampling_frequency=2, window_duration=2, slices=[(3, 1), (0, 0), (3, 1), (2, .5)])
    expected = torch.stack([data[:, 2:6, 3], data[:, :4, 0], data[:, 2:6, 3], data[:, 1:5, 2]], dim=-1)
    torch.testing.assert_close(step(data), expected)
    assert step.required_samples == 6
    assert step(data).dtype == data.dtype
    assert step(data).device == data.device


def test_single_input_channel_can_generate_multiple_delayed_outputs():
    data = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
    step = DelayedChannelSlices(sampling_frequency=1, window_duration=2, slices=[(0, 2), (0, 0)])
    torch.testing.assert_close(step(data), torch.tensor([[[2., 0.], [3., 1.]]]))


@pytest.mark.parametrize("kwargs, message", [
    ({"sampling_frequency": 0}, "sampling_frequency"),
    ({"sampling_frequency": float("nan")}, "sampling_frequency"),
    ({"window_duration": 0}, "window_duration"),
    ({"window_duration": .1}, "integral"),
    ({"slices": []}, "non-empty"),
    ({"slices": "0,0"}, "non-empty"),
    ({"slices": [[0]]}, "pair"),
    ({"slices": [[-1, 0]]}, "integer indices"),
    ({"slices": [[1.5, 0]]}, "integer indices"),
    ({"slices": [[True, 0]]}, "integer indices"),
    ({"slices": [[0, -1]]}, "non-negative"),
    ({"slices": [[0, float("inf")]]}, "non-negative"),
    ({"slices": [[0, .1]]}, "integral"),
])
def test_invalid_slice_configuration(kwargs, message):
    arguments = {"sampling_frequency": 64, "window_duration": 30, "slices": [(0, 0)]}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        DelayedChannelSlices(**arguments)


def test_slice_input_errors():
    step = DelayedChannelSlices(sampling_frequency=1, window_duration=2, slices=[(2, 1)])
    with pytest.raises(ValueError, match=r"\[B,T,C\]"):
        step(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="out of range"):
        step(torch.zeros(1, 3, 2))
    with pytest.raises(ValueError, match="at least 3 context samples"):
        step(torch.zeros(1, 2, 3))
    with pytest.raises(ValueError, match="finite"):
        step(torch.full((1, 3, 3), float("nan")))
    with pytest.raises(TypeError, match="floating-point"):
        step(torch.zeros(1, 3, 3, dtype=torch.long))


def test_scale_channels_multiple_amounts_explicit_clipping_and_input_immutability():
    data = torch.tensor([[[10., 50., 20., 500.], [20., 150., 40., 900.]]], dtype=torch.float64)
    before = data.clone()
    scale = ScaleChannels(scales={0: -2, 1: .01, 3: .001}, clip={1: [0, 1], 2: [0, 25]})
    expected = torch.tensor([[[-20., .5, 20., .5], [-40., 1., 25., .9]]], dtype=torch.float64)
    torch.testing.assert_close(scale(data), expected)
    torch.testing.assert_close(data, before)
    assert scale(data).dtype == data.dtype
    assert scale(data).device == data.device
    torch.testing.assert_close(ScaleChannels(scales={1: .01})(data)[..., 1], data[..., 1] * .01)
    torch.testing.assert_close(ScaleChannels(scales={}, clip={2: [0, 25]})(data)[..., 0], data[..., 0])


@pytest.mark.parametrize("kwargs, message", [
    ({"scales": {}}, "at least one"),
    ({"scales": {-1: 1}}, "integer indices"),
    ({"scales": {True: 1}}, "integer indices"),
    ({"scales": {"1": .01}}, "integer indices"),
    ({"scales": {0: float("nan")}}, "finite"),
    ({"scales": {0: float("inf")}}, "finite"),
    ({"scales": {0: 1}, "clip": {0: [2, 1]}}, "increasing"),
    ({"scales": {0: 1}, "clip": {0: [0, float("inf")]}}, "finite"),
    ({"scales": {0: 1}, "clip": {0: [0]}}, "minimum, maximum"),
    ({"scales": {0: 1}, "clip": {-1: [0, 1]}}, "integer indices"),
])
def test_invalid_scale_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ScaleChannels(**kwargs)


def test_scale_input_errors_and_mapping_types():
    with pytest.raises(TypeError, match="mappings"):
        ScaleChannels(scales=[.01])
    with pytest.raises(TypeError, match="mappings"):
        ScaleChannels(scales={0: 1}, clip=[0, 1])
    step = ScaleChannels(scales={2: .01})
    with pytest.raises(ValueError, match="out of range"):
        step(torch.zeros(1, 3, 2))
    with pytest.raises(ValueError, match="channel dimension"):
        step(torch.tensor(0.))
    with pytest.raises(ValueError, match="finite"):
        step(torch.full((1, 3, 3), float("nan")))
    with pytest.raises(TypeError, match="floating-point"):
        step(torch.zeros(1, 3, 3, dtype=torch.long))


def test_yaml_construction_warmup_serialization_and_gradient_composition():
    steps = build_value([
        {"name": "sleepwalker.models.preprocessors.ScaleChannels.ScaleChannels", "scales": {2: .01}, "clip": {2: [0, 1]}},
        {"name": "sleepwalker.models.preprocessors.DelayedChannelSlices.DelayedChannelSlices", "sampling_frequency": 1, "window_duration": 2, "slices": [[2, 1], [0, 0]]},
    ])
    data = torch.tensor([[[1., 9., 30.], [2., 9., 50.], [3., 9., 80.]]], requires_grad=True)
    pipeline = torch.nn.Sequential(*steps)
    for step in steps:
        assert not step.requires_warmup()
        step.update(data)
    expected = torch.tensor([[[.5, 1.], [.8, 2.]]])
    torch.testing.assert_close(pipeline(data), expected)
    restored = cloudpickle.loads(cloudpickle.dumps(pipeline))
    torch.testing.assert_close(restored(data), expected)
    pipeline(data).sum().backward()
    torch.testing.assert_close(data.grad, torch.tensor([[[1., 0., 0.], [1., 0., .01], [0., 0., .01]]]))
