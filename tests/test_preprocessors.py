import os
from pathlib import Path
import pytest
import torch

from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.SyntheticDataset import SyntheticDataset
from sleepwalker.datasets.utils import get_edf_files_in_repo
from sleepwalker.models.preprocessors.Normalize import Normalize
from sleepwalker.models.preprocessors.Spectogram import Spectogram

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


# --------------------------
# Spectogram tests
# --------------------------

def test_spectrogram_basic_shape():
    torch.manual_seed(0)
    spec = Spectogram(n_fft=64, hop_length=16)
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
    spec = Spectogram(n_fft=64, hop_length=32, epoch_len_samples=128)
    data = torch.randn(2, 512, 1)  # B, T, D

    out = spec(data)
    # B * (T // epoch_len_samples) = 2 * 4 = 8
    B, F, T_, D = out.shape
    assert B == 8
    assert D == 1
    assert torch.isfinite(out).all()


def test_spectrogram_with_custom_win_length():
    torch.manual_seed(0)
    spec = Spectogram(n_fft=128, hop_length=32, win_length=64)
    data = torch.randn(1, 512, 2)
    out = spec(data)
    assert out.shape[-1] == 2  # D
    assert torch.isfinite(out).all()


def test_spectrogram_consistency_cpu_vs_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)

    spec_cpu = Spectogram(n_fft=64, hop_length=16)
    spec_gpu = Spectogram(n_fft=64, hop_length=16)

    data = torch.randn(1, 256, 2)
    out_cpu = spec_cpu(data)
    out_gpu = spec_gpu(data.to("cuda")).cpu()

    # They should be numerically close
    assert torch.allclose(out_cpu, out_gpu, atol=1e-5, rtol=1e-3)

def test_peprocessor_chain(device="cuda"):
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    NUM_PATIENTS = int(os.environ.get("NUM_PATIENTS", 5))

    edf_data_dir = os.path.join(Path(__file__).parent, "data")
    
    edf_files = get_edf_files_in_repo(edf_data_dir, recursive=True)
    assert len(edf_files) > 0
    edf_files = edf_files[:NUM_PATIENTS]
    preprocessors = [Spectogram(), Normalize()]
    dataset = SyntheticDataset(total_input="120s", patients = edf_files, channels = [ChannelConfig(name="EEG", normalizer=None)], sample_frequency=100, event_mapping={}, remove_unmapped_events=False)
    iterate_dataset(dataset, NUM_BATCHES, preprocessors=preprocessors, device=device)  

if __name__ == '__main__':
    logger.context("CUDA TEST")
    test_peprocessor_chain("cuda")
    logger.uncontext()

    logger.context("CPU TEST")
    test_peprocessor_chain("cpu")
    logger.uncontext()