import os
from pathlib import Path
import pytest
import torch

from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.SyntheticDataset import SyntheticDataset
from sleepwalker.datasets.utils import get_edf_files_in_repo
from sleepwalker.models.preprocessors.Crop import Crop
from sleepwalker.models.preprocessors.EmpiricalClipScaler import EmpiricalClipScaler
from sleepwalker.models.preprocessors.Normalize import Normalize
from sleepwalker.models.preprocessors.NormalizeAlongDim import NormalizeAlongDim
from sleepwalker.models.preprocessors.RobustScaler import RobustScaler
from sleepwalker.models.preprocessors.Spectrogram import Spectrogram

from sleepwalker.utils import logger 

from dotenv import load_dotenv

from tests.utils import iterate_dataset
load_dotenv()  


# --------------------------
# Normalize tests
# --------------------------

def test_normalize_warmup_and_call():
    torch.manual_seed(0)
    norm = Normalize()
    assert norm.requires_warmup()

    data = torch.randn(100, 4)
    norm.update(data)

    # mean and M2 must be initialized
    assert norm.mean is not None
    assert norm.M2 is not None
    assert norm.count == 100

    # After call, output should be roughly zero mean and unit std
    out = norm(data)
    assert out.shape == data.shape
    assert torch.allclose(out.mean(0), torch.zeros_like(out.mean(0)), atol=1e-1)
    assert torch.allclose(out.std(0), torch.ones_like(out.std(0)), atol=1e-1)


def test_normalize_incremental_update():
    torch.manual_seed(0)
    norm = Normalize()
    data1 = torch.randn(50, 2)
    data2 = torch.randn(50, 2)

    norm.update(data1)
    mean1 = norm.mean.clone()
    count1 = norm.count

    norm.update(data2)
    # count should have increased
    assert norm.count == count1 + 50
    assert not torch.allclose(norm.mean, mean1)  # mean should change

    out = norm(torch.cat([data1, data2], dim=0))
    assert torch.isfinite(out).all()


def test_normalize_no_update_call_does_nothing():
    data = torch.randn(10, 3)
    norm = Normalize()
    out = norm(data)
    # Should be identical since mean and var are None
    assert torch.allclose(out, data)


def test_normalize_stability_zero_variance():
    data = torch.ones(10, 3)
    norm = Normalize()
    norm.update(data)
    out = norm(data)
    # No NaNs or infs expected
    assert torch.isfinite(out).all()
    # Zero variance → output should be all zeros
    assert torch.allclose(out, torch.zeros_like(out))


def test_normalize_selective_channels_only_updates_requested_indices():
    data = torch.tensor(
        [
            [[1.0, 10.0, 100.0], [3.0, 10.0, 200.0]],
            [[2.0, 10.0, 150.0], [4.0, 10.0, 300.0]],
        ]
    )
    norm = Normalize(channels=[0, 2])
    norm.update(data)

    out = norm.transform(data)
    assert out.shape == data.shape
    assert torch.allclose(out[..., 1], data[..., 1])
    assert not torch.allclose(out[..., 0], data[..., 0])
    assert not torch.allclose(out[..., 2], data[..., 2])


@pytest.mark.parametrize(
    "channels,pattern",
    [
        ([], "must not be empty"),
        ([0, 0], "duplicate"),
        ([-1], "negative"),
    ],
)
def test_preprocessor_rejects_invalid_channel_configs(channels, pattern):
    with pytest.raises(ValueError, match=pattern):
        Normalize(channels=channels)


def test_preprocessor_rejects_out_of_range_channels_at_runtime():
    norm = Normalize(channels=[0, 3])
    data = torch.randn(2, 4, 3)
    with pytest.raises(ValueError, match="out of range"):
        norm.transform(data)


# --------------------------
# Spectogram tests
# --------------------------

def test_spectrogram_basic_shape():
    torch.manual_seed(0)
    spec = Spectrogram(n_fft=64, hop_length=16)
    assert not spec.requires_warmup()

    data = torch.randn(2, 512, 3)  # B, T, D
    out = spec(data)
    assert out.ndim == 4  # (B, F, T', D)
    B, F, T_, D = out.shape
    assert B == 2 and D == 3
    assert F == 33  # n_fft//2 + 1
    assert T_ > 0
    assert torch.isfinite(out).all()


def test_spectrogram_with_epoch_len():
    torch.manual_seed(0)
    spec = Spectrogram(n_fft=64, hop_length=32, epoch_len_samples=128)
    data = torch.randn(2, 512, 1)  # B, T, D

    out = spec(data)
    # B * (T // epoch_len_samples) = 2 * 4 = 8
    B, F, T_, D = out.shape
    assert B == 8
    assert D == 1
    assert torch.isfinite(out).all()


def test_spectrogram_with_custom_win_length():
    torch.manual_seed(0)
    spec = Spectrogram(n_fft=128, hop_length=32, win_length=64)
    data = torch.randn(1, 512, 2)
    out = spec(data)
    assert out.shape[-1] == 2  # D
    assert torch.isfinite(out).all()


def test_spectrogram_consistency_cpu_vs_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)

    spec_cpu = Spectrogram(n_fft=64, hop_length=16)
    spec_gpu = Spectrogram(n_fft=64, hop_length=16)

    data = torch.randn(1, 256, 2)
    out_cpu = spec_cpu(data)
    out_gpu = spec_gpu(data.to("cuda")).cpu()

    # They should be numerically close
    assert torch.allclose(out_cpu, out_gpu, atol=1e-5, rtol=1e-3)


def test_spectrogram_rejects_selective_channel_merge_when_output_shape_changes():
    spec = Spectrogram(n_fft=64, hop_length=16, channels=[0])
    data = torch.randn(2, 512, 3)
    with pytest.raises(ValueError, match="Selective preprocessing requires"):
        spec.transform(data)

# --------------------------
# Crop tests
# --------------------------

def test_crop_middle():
    data = torch.arange(2 * 10 * 3).reshape(2, 10, 3)
    crop = Crop(total_input="4s", sampling_rate="1s", where="middle")
    out = crop(data)

    assert out.shape == (2, 4, 3)
    expected = data[:, 3:7, :]
    assert torch.equal(out, expected)


def test_crop_left():
    data = torch.arange(2 * 10 * 3).reshape(2, 10, 3)
    crop = Crop(total_input="4s", sampling_rate="1s", where="left")
    out = crop(data)

    assert out.shape == (2, 4, 3)
    expected = data[:, 0:4, :]
    assert torch.equal(out, expected)


def test_crop_right():
    data = torch.arange(2 * 10 * 3).reshape(2, 10, 3)
    crop = Crop(total_input="4s", sampling_rate="1s", where="right")
    out = crop(data)

    assert out.shape == (2, 4, 3)
    expected = data[:, 6:10, :]
    assert torch.equal(out, expected)


def test_crop_raises_if_too_long():
    data = torch.arange(2 * 10 * 3).reshape(2, 10, 3)
    crop = Crop(total_input="12s", sampling_rate="1s")

    with pytest.raises(ValueError, match="exceeds input length"):
        _ = crop(data)


def test_crop_invalid_where():
    data = torch.arange(2 * 10 * 3).reshape(2, 10, 3)
    crop = Crop(total_input="2s", sampling_rate="1s", where="invalid")

    with pytest.raises(ValueError, match="Invalid crop location"):
        _ = crop(data)


def test_crop_len_computation():
    crop = Crop(total_input="5s", sampling_rate="0.5s")
    assert crop.len == 10  # 5 / 0.5 = 10


def test_crop_requires_warmup_false():
    crop = Crop(total_input="1s", sampling_rate="1s")
    assert crop.requires_warmup() is False

# --------------------------
# EmpiricalClipScaler tests
# --------------------------

def test_ecs_requires_warmup_true():
    s = EmpiricalClipScaler()
    assert s.requires_warmup() is True


def test_ecs_update_sets_mins_maxs():
    data = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (B=1, T=3, D=2)
    s = EmpiricalClipScaler(q=0.8, scale=1)

    assert s.mins is None
    s.update(data)
    assert s.mins is not None
    assert s.maxs is not None
    assert s.mins.shape == (2,)
    assert s.maxs.shape == (2,)


def test_ecs_update_accumulates_extremes():
    s = EmpiricalClipScaler(q=0.8, scale=1)

    d1 = torch.tensor([[[0.0, 10.0], [1.0, 20.0]]])  # smaller range
    d2 = torch.tensor([[[5.0, 5.0], [15.0, 25.0]]])  # larger range

    s.update(d1)
    mins_before, maxs_before = s.mins.clone(), s.maxs.clone()
    s.update(d2)
    # mins should become smaller or equal, maxs larger or equal
    assert torch.all(s.mins <= mins_before)
    assert torch.all(s.maxs >= maxs_before)


def test_ecs_call_scales_between_0_and_1():
    s = EmpiricalClipScaler()
    s.mins = torch.tensor([0.0, 10.0])
    s.maxs = torch.tensor([10.0, 20.0])

    data = torch.tensor([[[5.0, 15.0], [10.0, 30.0]]])  # (1, 2, 2)
    out = s(data)

    # Values should be clamped and scaled into [0,1]
    assert out.shape == data.shape
    assert torch.all((out >= 0) & (out <= 1))

    # Test that the middle value maps to roughly 0.5
    assert torch.isclose(out[0, 0, 0], torch.tensor(0.5), atol=1e-5)
    assert torch.isclose(out[0, 0, 1], torch.tensor(0.5), atol=1e-5)


def test_ecs_call_without_update_returns_input(monkeypatch):
    s = EmpiricalClipScaler()
    data = torch.randn(1, 5, 2)
    out = s(data)

    # Should return unchanged tensor
    assert torch.equal(out, data)

def test_ecs_call_handles_zero_denom():
    s = EmpiricalClipScaler()
    s.mins = torch.tensor([1.0, 1.0])
    s.maxs = torch.tensor([1.0, 1.0])  # zero denominator

    data = torch.tensor([[[1.0, 1.0]]])
    out = s(data)

    # Avoids division by zero, so output should be 0
    assert torch.allclose(out, torch.zeros_like(out))

# --------------------------
# RobustScaler tests
# --------------------------

def test_rs_requires_warmup_true():
    s = RobustScaler()
    assert s.requires_warmup() is True

def test_rs_push_initializes_correctly():
    s = RobustScaler()
    data = torch.randn(2, 4, 3)  # (B=2, T=4, D=3)
    assert not s.is_initialized

    s.push(data)

    # After first push, internal structures should be initialized
    assert s.is_initialized
    assert hasattr(s, "marker_heights")
    assert s.marker_heights.shape == (3, 5)
    assert torch.all(s.initialized)  # each feature initialized

def test_rs_push_updates_marker_extremes():
    s = RobustScaler()
    data = torch.tensor([[[0.0, 10.0], [1.0, 20.0]]])
    s.push(data)

    before = s.marker_heights.clone()
    # Push larger data -> max marker should increase
    s.push(torch.tensor([[[5.0, 50.0], [10.0, 100.0]]]))
    assert torch.all(s.marker_heights[:, 4] >= before[:, 4])
    assert torch.all(s.marker_heights[:, 0] <= before[:, 0])

def test_rs_call_scales_data_when_initialized():
    s = RobustScaler()
    data = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    s.push(data)  # initializes with 2 features

    out = s(data)
    assert out.shape == data.shape
    assert torch.isfinite(out).all()  # no NaNs or infs

def test_rs_call_returns_input_when_uninitialized():
    s = RobustScaler()
    data = torch.randn(1, 5, 2)
    out = s(data)
    assert torch.equal(out, data)

def test_rs_iqr_zero_handling():
    s = RobustScaler()
    # Manually set marker_heights to zero IQR
    s.is_initialized = True
    s.marker_heights = torch.tensor([
        [1.0, 1.0, 2.0, 1.0, 3.0],
        [2.0, 2.0, 2.0, 2.0, 2.0]
    ])
    data = torch.tensor([[[1.0, 2.0], [2.0, 2.0]]])
    out = s(data)

    # Should not produce NaNs or infs
    assert torch.isfinite(out).all()

# --------------------------
# NormalizeAlongDim tests
# --------------------------

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_normalizealongdim_ctor(device):
    """Ensure NormalizeAlongDim constructor works on CPU and CUDA."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    norm = NormalizeAlongDim(dim=1)
    assert isinstance(norm, NormalizeAlongDim)
    assert norm.requires_warmup() is False

@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_normalizealongdim_forward(device, dim):
    """Run one forward normalization pass along different dimensions."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # shape: (B, T, D)
    B, T, D = 4, 10, 3
    x = torch.randn(B, T, D, device=device)

    norm = NormalizeAlongDim(dim=dim)
    with torch.no_grad():
        y = norm(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    # Check zero mean and unit variance along chosen dim (roughly)
    mean = y.mean(dim=dim)
    std = y.std(dim=dim)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-3)

def test_peprocessor_chain(device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    NUM_PATIENTS = int(os.environ.get("NUM_PATIENTS", 5))

    edf_data_dir = os.path.join(Path(__file__).parent, "data")
    
    edf_files = get_edf_files_in_repo(edf_data_dir, recursive=True)
    assert len(edf_files) > 0
    edf_files = edf_files[:NUM_PATIENTS]
    preprocessors = [Spectrogram(), Normalize()]
    dataset = SyntheticDataset(
        total_input="120s",
        channels=[ChannelConfig(name="EEG", normalizer=None)],
        sample_frequency=100,
        event_mapping={},
        remove_unmapped_events=False,
    )
    dataset.initialize(edf_files)
    iterate_dataset(dataset, NUM_BATCHES, preprocessors=preprocessors, device=device)  

if __name__ == '__main__':
    logger.context("CUDA TEST")
    test_peprocessor_chain("cuda")
    logger.uncontext()

    logger.context("CPU TEST")
    test_peprocessor_chain("cpu")
    logger.uncontext()
