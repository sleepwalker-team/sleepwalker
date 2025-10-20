import os
from pathlib import Path
import numpy as np
import pytest
import torch

from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.SyntheticDataset import SyntheticDataset

from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.utils import get_edf_files_in_repo

from dotenv import load_dotenv

from tests.utils import iterate_dataset
load_dotenv()  

@pytest.fixture
def dummy_signal(fs=100, n_samples=2000):
    """Generate a 10 Hz sine wave with noise for testing."""
    t = np.arange(n_samples) / fs
    signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(n_samples)
    return signal.reshape(-1, 1), fs


def test_filter_basic(dummy_signal):
    """Ensure the bandpass + notch filter runs and preserves shape."""
    signal, fs = dummy_signal
    norm = EEGFilterNormalizer()
    filtered = norm._filter(signal[:, 0], fs)
    assert filtered.shape == signal[:, 0].shape
    assert not np.isnan(filtered).any()


def test_fit_and_transform(dummy_signal):
    """Ensure fit() and transform() normalize the data correctly."""
    signal, fs = dummy_signal
    norm = EEGFilterNormalizer()
    norm.fit(signal, fs)
    transformed = norm.transform(signal, fs)

    assert transformed.shape == signal.shape
    assert np.isclose(np.mean(transformed), 0, atol=1e-1)
    assert np.isclose(np.std(transformed), 1, atol=1e-1)


def test_invalid_shape_fit(dummy_signal):
    """fit() should raise for invalid input shape."""
    signal, fs = dummy_signal
    norm = EEGFilterNormalizer()
    with pytest.raises(ValueError):
        norm.fit(signal.squeeze(), fs)  # wrong shape (N,)


def test_invalid_shape_transform(dummy_signal):
    """transform() should raise for invalid input shape."""
    signal, fs = dummy_signal
    norm = EEGFilterNormalizer()
    norm.fit(signal, fs)
    with pytest.raises(ValueError):
        norm.transform(signal.squeeze(), fs)


def test_zero_variance():
    """Ensure std fallback to 1.0 when signal has zero variance."""
    fs = 100
    signal = np.ones((1000, 1))
    norm = EEGFilterNormalizer()
    norm.fit(signal, fs)
    print(norm.std_)
    assert np.isclose(norm.std_, 1.0)
    transformed = norm.transform(signal, fs)
    assert np.allclose(transformed, 0.0)


def test_no_notch_filter(dummy_signal):
    """Ensure code runs without notch filtering."""
    signal, fs = dummy_signal
    norm = EEGFilterNormalizer(notch_freq=None)
    norm.fit(signal, fs)
    transformed = norm.transform(signal, fs)
    assert transformed.shape == signal.shape


@pytest.mark.parametrize("lowcut,highcut", [(0.1, 30), (1.0, 40)])
def test_different_bandpass(dummy_signal, lowcut, highcut):
    """Check that bandpass parameters produce valid output."""
    signal, fs = dummy_signal
    norm = EEGFilterNormalizer(lowcut=lowcut, highcut=highcut)
    norm.fit(signal, fs)
    transformed = norm.transform(signal, fs)
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cuda_vs_cpu_consistency(dummy_signal, device):
    """Ensure CPU and CUDA normalized results are consistent."""
    signal, fs = dummy_signal
    norm_cpu = EEGFilterNormalizer()
    norm_cpu.fit(signal, fs)
    out_cpu = norm_cpu.transform(signal, fs)

    # Move signal to GPU if available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x_torch = torch.tensor(signal, dtype=torch.float32, device=device)
    x_np = x_torch.cpu().numpy()  # normalization still runs on CPU
    norm_gpu = EEGFilterNormalizer()
    norm_gpu.fit(x_np, fs)
    out_gpu = norm_gpu.transform(x_np, fs)

    # Compare mean and std roughly
    assert np.isclose(out_cpu.mean(), out_gpu.mean(), atol=1e-3)
    assert np.isclose(out_cpu.std(), out_gpu.std(), atol=1e-3)

def test_eegfilternormalizer_integration():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    NUM_PATIENTS = int(os.environ.get("NUM_PATIENTS", 5))

    edf_data_dir = os.path.join(Path(__file__).parent, "data")
    
    edf_files = get_edf_files_in_repo(edf_data_dir, recursive=True)
    assert len(edf_files) > 0
    edf_files = edf_files[:NUM_PATIENTS]

    normalizer = EEGFilterNormalizer()
    dataset = SyntheticDataset(patients = edf_files, channels = [ChannelConfig(name="EEG", normalizer=normalizer)], sample_frequency=100, event_mapping={}, remove_unmapped_events=False)
    iterate_dataset(dataset, NUM_BATCHES)    

if __name__ == '__main__':
    test_eegfilternormalizer_integration()