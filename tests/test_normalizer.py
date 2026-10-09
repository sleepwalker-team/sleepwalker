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

@pytest.mark.parametrize("fs,n_samples", [(100, 2000), (200, 4000)])
def test_eegfilternorm_filter_basic(fs, n_samples):
    """Ensure the bandpass + notch filter runs and preserves shape."""
    t = np.arange(n_samples) / fs
    signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(n_samples)
    norm = EEGFilterNormalizer(fs=fs)
    filtered = norm.filter(signal)
    assert filtered.shape == signal.shape
    assert not np.isnan(filtered).any()


@pytest.mark.parametrize("fs,n_samples", [(100, 2000), (200, 4000)])
def test_eegfilternorm_fixed_distribution_and_transform(fs, n_samples):
    """Ensure fixed mean/std values normalize filtered data correctly."""
    t = np.arange(n_samples) / fs
    signal = (np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(n_samples)).reshape(-1, 1)

    filter_only = EEGFilterNormalizer(fs=fs, normalize=False)
    filtered = filter_only(signal, unit="uV", is_recording=False)[0]
    norm = EEGFilterNormalizer(fs=fs, mean=float(filtered.mean()), std=float(filtered.std()))
    transformed = norm(signal, unit="uV", is_recording=False)[0]

    assert transformed.shape == signal.shape
    assert np.isclose(np.mean(transformed), 0, atol=1e-1)
    assert np.isclose(np.std(transformed), 1, atol=1e-1)


def test_eegfilternorm_can_filter_without_patient_normalization():
    fs = 100
    t = np.arange(2000) / fs
    signal = (3.0 + np.sin(2 * np.pi * 10 * t)).reshape(-1, 1)
    norm = EEGFilterNormalizer(fs=fs, normalize=False)

    transformed = norm(signal, unit="uV", is_recording=False)[0]

    assert np.allclose(transformed[:, 0], norm.filter(signal[:, 0]))


@pytest.mark.parametrize("fs", [100, 200])
def test_eegfilternorm_invalid_shape_transform(fs):
    """transform() should raise for invalid input shape."""
    t = np.arange(2000) / fs
    signal = (np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))).reshape(-1, 1)
    norm = EEGFilterNormalizer(fs=fs)
    with pytest.raises(ValueError):
        norm(signal.squeeze(), unit="uV", is_recording=False)[0]


def test_eegfilternorm_rejects_nonpositive_std():
    with pytest.raises(ValueError, match="std must be finite and positive"):
        EEGFilterNormalizer(fs=100, std=0)

@pytest.mark.parametrize("lowcut,highcut,fs", [(0.1, 30, 100), (1.0, 40, 200)])
def test_eegfilternorm_different_bandpass(lowcut, highcut, fs):
    """Check that bandpass parameters produce valid output."""
    t = np.arange(2000) / fs
    signal = (np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))).reshape(-1, 1)
    norm = EEGFilterNormalizer(fs=fs, lowcut=lowcut, highcut=highcut)
    transformed = norm(signal, unit="uV", is_recording=False)[0]
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize("device,fs", [("cpu", 100), ("cuda", 200)])
def test_eegfilternorm_cuda_vs_cpu_consistency(device, fs):
    """Ensure CPU and CUDA normalized results are consistent."""
    n_samples = 2000
    t = np.arange(n_samples) / fs
    signal = (np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(n_samples)).reshape(-1, 1)

    norm_cpu = EEGFilterNormalizer(fs=fs)
    out_cpu = norm_cpu(signal, unit="uV", is_recording=False)[0]

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x_torch = torch.tensor(signal, dtype=torch.float32, device=device)
    x_np = x_torch.cpu().numpy()
    norm_gpu = EEGFilterNormalizer(fs=fs)
    out_gpu = norm_gpu(x_np, unit="uV", is_recording=False)[0]

    assert np.isclose(out_cpu.mean(), out_gpu.mean(), atol=1e-3)
    assert np.isclose(out_cpu.std(), out_gpu.std(), atol=1e-3)


def test_eegfilternorm_integration():
    """Integration test with synthetic dataset + EDF files."""
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    NUM_PATIENTS = int(os.environ.get("NUM_PATIENTS", 5))
    edf_data_dir = os.path.join(Path(__file__).parent, "data")

    edf_files = get_edf_files_in_repo(edf_data_dir, recursive=True)
    assert len(edf_files) > 0
    edf_files = edf_files[:NUM_PATIENTS]

    fs = 100
    normalizer = EEGFilterNormalizer(fs=fs)
    dataset = SyntheticDataset(
        channels=[ChannelConfig("EEG", ["EEG"], preprocessors=[] if normalizer is None else [normalizer])],
        sample_frequency=fs,
        event_mapping={},
        remove_unmapped_events=False,
    )
    dataset.initialize(edf_files)
    iterate_dataset(dataset, NUM_BATCHES)

if __name__ == '__main__':
    test_eegfilternorm_integration()
