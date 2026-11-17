"""Checkpoint consent/integrity checks and opt-in released-weight integration tests."""

import hashlib
import importlib
import io
import os
from pathlib import Path
import urllib.error

import pytest
import torch

from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment import load_packaged_model, save_packaged_model
from sleepwalker.models import OSF, SleepFM
from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel
from sleepwalker.models.utils import resolve_checkpoint
from sleepwalker.models.PackagedClassifierModel import PackagedClassifierModel
from sleepwalker.models.PackagedSequenceClassifierModel import PackagedSequenceClassifierModel


class InterruptedTransfer(io.BytesIO):
    def read(self, size=-1):
        chunk = super().read(size)
        if not chunk:
            raise urllib.error.URLError("interrupted transfer")
        return chunk


@pytest.mark.parametrize("model_name", ["sleepfm", "osf"])
@pytest.mark.parametrize("use_default_path", [False, True])
def test_checkpoint_download_requires_consent_and_reuses_verified_file(tmp_path, monkeypatch, model_name, use_default_path):
    content = b"released weights"
    digest = hashlib.sha256(content).hexdigest()
    checkpoint = None if use_default_path else tmp_path / "weights" / "released.pt"
    options = {"url": "https://example.com/released.pt", "sha256": digest, "checkpoint": checkpoint}
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    requests = []

    def download(url, *, timeout):
        requests.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr("urllib.request.urlopen", download)
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        resolve_checkpoint(model_name, **options)
    assert requests == []
    assert list(tmp_path.iterdir()) == []
    path = resolve_checkpoint(model_name, **options, allow_download=True)
    expected_path = tmp_path / ".cache" / "sleepwalker" / "foundation" / model_name / f"{digest}.pt" if use_default_path else checkpoint
    assert path == expected_path
    assert path.read_bytes() == content
    assert resolve_checkpoint(model_name, **options) == path
    assert resolve_checkpoint(model_name, **options, allow_download=True) == path
    assert len(requests) == 1
    path.write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        resolve_checkpoint(model_name, **options, allow_download=True)
    assert len(requests) == 1


def test_existing_checkpoint_is_verified_without_downloading(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("An existing checkpoint attempted a download.")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    content = b"released weights"
    digest = hashlib.sha256(content).hexdigest()
    options = {"url": "https://example.com/released.pt", "sha256": digest}
    checkpoint = tmp_path / "released.pt"
    checkpoint.write_bytes(content)
    assert resolve_checkpoint("sleepfm", **options, checkpoint=checkpoint) == checkpoint
    assert resolve_checkpoint("sleepfm", **options, checkpoint=checkpoint, allow_download=True) == checkpoint
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        resolve_checkpoint("sleepfm", **options, checkpoint=checkpoint, allow_download=True)
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("sleepfm", **options, checkpoint=tmp_path / "missing.pt")


@pytest.mark.parametrize("failure", ["checksum", "transfer"])
def test_failed_download_does_not_publish_a_checkpoint(tmp_path, monkeypatch, failure):
    def download(url, *, timeout):
        if failure == "transfer":
            return InterruptedTransfer(b"partial checkpoint")
        return io.BytesIO(b"invalid checkpoint")

    monkeypatch.setattr("urllib.request.urlopen", download)
    with pytest.raises(ValueError if failure == "checksum" else urllib.error.URLError):
        resolve_checkpoint("osf", url="https://example.com/released.pt", sha256=hashlib.sha256(b"released weights").hexdigest(), checkpoint=tmp_path / "weights" / "released.pt", allow_download=True)
    assert list(tmp_path.rglob("*.part")) == []
    assert list(tmp_path.rglob("*.pt")) == []


def test_invalid_window_fails_before_loading_weights(tmp_path):
    with pytest.raises(ValueError, match="multiple of 640"):
        SleepFM(checkpoint=tmp_path / "released.pt", allow_download=True, ts_len=641)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("model_class", [SleepFM, OSF])
def test_models_initialize_without_external_weights_and_use_standard_model_interfaces(tmp_path, monkeypatch, model_class):
    def refuse(*args, **kwargs):
        raise AssertionError("Random model initialization attempted to load external weights.")

    monkeypatch.setattr(importlib.import_module(model_class.__module__), "resolve_checkpoint", refuse)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = model_class(**({"ts_len": 640} if model_class is SleepFM else {}))
        assert isinstance(model, (BaseModel, EmbeddingModel))
        assert model.training
        assert model.checkpoint_path is None
        assert list(tmp_path.iterdir()) == []
        model.eval()
        x = torch.randn(2, model.ts_len, model.n_channels)
        with torch.inference_mode():
            torch.testing.assert_close(model(x), model.features(x))
        assert model.feature_dim() == model.features(x[:1]).shape[1]
    finally:
        torch.set_num_threads(original_threads)


@pytest.mark.parametrize("model_class", [SleepFM, OSF])
@pytest.mark.parametrize("options", [{"checkpoint": "released.pt"}, {"allow_download": True}, {"checkpoint": "released.pt", "allow_download": True}])
def test_each_weight_option_requests_a_checkpoint_before_constructing_the_encoder(monkeypatch, model_class, options):
    def missing(model_name, **kwargs):
        assert kwargs["checkpoint"] == options.get("checkpoint")
        assert kwargs["allow_download"] == options.get("allow_download", False)
        raise FileNotFoundError("requested checkpoint")

    monkeypatch.setattr(importlib.import_module(model_class.__module__), "resolve_checkpoint", missing)
    with pytest.raises(FileNotFoundError, match="requested checkpoint"):
        model_class(**options)


@pytest.mark.parametrize("model_name,expected_shape", [("sleepfm", (2, 30720)), ("osf", (2, 768))])
def test_released_model_package_loads_without_checkpoint_and_trains_a_head(tmp_path, monkeypatch, model_name, expected_shape):
    checkpoint = os.environ.get(f"SLEEPWALKER_{model_name.upper()}_CHECKPOINT")
    if checkpoint is None:
        pytest.skip(f"Set SLEEPWALKER_{model_name.upper()}_CHECKPOINT to check released weights.")
    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        def refuse_network(*args, **kwargs):
            raise AssertionError("Local released weights attempted a download.")

        monkeypatch.setattr("urllib.request.urlopen", refuse_network)
        model = (SleepFM if model_name == "sleepfm" else OSF)(checkpoint=checkpoint)
        assert not model.training
        assert all(parameter.requires_grad for parameter in model.parameters())
        duration = f"{model.ts_len / model.sampling_frequency}s"
        dataset = UnlabelledDataset(channels=[ChannelConfig(name, [name]) for name in model.input_channels], sample_frequency=model.sampling_frequency, total_input=duration, stride=duration, resample_type="polyphase" if model_name == "sleepfm" else "nearest", z_normalize=True)
        assert dataset.get_timeseries_len() == model.ts_len
        if model_name == "osf":
            assert dataset.get_input_channels() == ["ECG", "EMG_Chin", "EMG_LLeg", "EMG_RLeg", "ABD", "THX", "NP", "SN", "EOG_E1_A2", "EOG_E2_A1", "EEG_C3_A2", "EEG_C4_A1"]
        torch.manual_seed(4)
        x = torch.randn(2, model.ts_len, model.n_channels)
        with torch.inference_mode():
            expected = model.features(x)
        assert tuple(expected.shape) == expected_shape
        assert torch.isfinite(expected).all()
        with pytest.raises(ValueError, match="Expected input shaped"):
            model.features(x.transpose(1, 2))
        save_packaged_model(tmp_path / "package", name=model_name, model=model, dataset=dataset, config={"source": model.source})
        # The package must load without consulting the original checkpoint or network.
        def refuse(*args, **kwargs):
            raise AssertionError("Package load attempted to access external weights.")

        monkeypatch.setattr(importlib.import_module("sleepwalker.models.SleepFM"), "resolve_checkpoint", refuse)
        monkeypatch.setattr(importlib.import_module("sleepwalker.models.OSF"), "resolve_checkpoint", refuse)
        monkeypatch.setattr("urllib.request.urlopen", refuse)
        loaded = load_packaged_model(tmp_path / "package")
        loaded.model.checkpoint_path = Path("/missing/released-checkpoint.pt")
        with torch.inference_mode():
            torch.testing.assert_close(loaded.model.features(x), expected, atol=1e-5, rtol=1e-4)
        head = PackagedClassifierModel(package=loaded, classes=["a", "b"], ts_len=model.ts_len, freeze_encoder=True)
        head.train()
        head(x).sum().backward()
        assert head.head.weight.grad is not None
        assert all(parameter.grad is None for parameter in head.encoder.parameters())
        if model_name == "sleepfm":
            sequence = PackagedSequenceClassifierModel(package=loaded, classes=["a", "b"], ts_len=model.ts_len, sequence_len=10, token_count=60, token_dim=128, modality_count=4)
            assert sequence(x).shape == (2, 10, 2)
        # Moving the native model must also move every buffer used by its forward pass.
        loaded.model.to(device="meta")
        assert loaded.model.features(torch.empty_like(x, device="meta")).shape == expected_shape
    finally:
        torch.set_num_threads(original_threads)
