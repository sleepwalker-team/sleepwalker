import pytest
import torch


from dotenv import load_dotenv

from sleepwalker.models.AttnSleep import AttnSleep
from sleepwalker.models.MRASleepNet import MRASleepNet
from sleepwalker.models.SeqSleepNet import SeqSleepNet
from sleepwalker.models.SleepTransformer import SleepTransformer
load_dotenv()  

# --------------------------
# SleepTransformer tests
# --------------------------

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sleeptransformer_ctor(device):
    """Ensure SleepTransformer constructor runs on both CPU and CUDA."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = SleepTransformer(
        classes=["W", "N1", "N2", "N3", "REM"],
        n_channels=5,
    ).to(device)

    assert isinstance(model, SleepTransformer)
    for p in model.parameters():
        assert p.device.type == device

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sleeptransformer_forward(device):
    """Run one forward pass through SleepTransformer on CPU and CUDA."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, T, D = 2, 630_000, 5  # batch, time steps, channels
    x = torch.randn(B, T, D, device=device)

    model = SleepTransformer(
        classes=["W", "N1", "N2", "N3", "REM"],
        n_channels=D,
    ).to(device)

    with torch.no_grad():
        y = model(x)

    assert torch.isfinite(y).all()
    assert y.ndim == 2  # (B, C)
    assert y.shape[0] == B
    assert y.shape[1] == len(["W", "N1", "N2", "N3", "REM"])

# --------------------------
# AttnSleep tests
# --------------------------

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_attnsleep_ctor(device):
    """Ensure AttnSleep constructor runs on both CPU and CUDA."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = AttnSleep(
        n_channels=5,
        ts_len=30_000,
        classes=["W", "N1", "N2", "N3", "REM"],
    ).to(device)

    assert isinstance(model, AttnSleep)
    for p in model.parameters():
        assert p.device.type == device

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_attnsleep_forward(device):
    """Run one forward pass through AttnSleep on CPU and CUDA."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, T, D = 2, 30_000, 5  # batch, time steps, channels
    x = torch.randn(B, T, D, device=device)

    model = AttnSleep(
        n_channels=D,
        ts_len=T,
        classes=["W", "N1", "N2", "N3", "REM"],
    ).to(device)

    with torch.no_grad():
        y = model(x)

    # sanity checks
    assert torch.isfinite(y).all()
    assert y.ndim in (2, 3)
    if y.ndim == 2:
        # classification output (B, C)
        assert y.shape == (B, 5)
    else:
        # sequence output (B, L, C)
        assert y.shape[0] == B
        assert y.shape[-1] == 5

# --------------------------
# MRASleepNet tests
# --------------------------

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mrasleepnet_ctor(device):
    """Ensure MRASleepNet constructor runs on both CPU and CUDA."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = MRASleepNet(ts_len=3000, n_features=5, classes=["W", "N1", "N2", "N3", "REM"]).to(device)
    assert isinstance(model, MRASleepNet)
    for p in model.parameters():
        assert p.device.type == device


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mrasleepnet_forward(device):
    """Run one forward pass through MRASleepNet on CPU and CUDA."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, T, D = 2, 3000, 5  # batch, time, features
    x = torch.randn(B, T, D, device=device)

    model = MRASleepNet(ts_len=T, n_features=D, classes=["W", "N1", "N2", "N3", "REM"]).to(device)

    with torch.no_grad():
        y = model(x)

    assert torch.isfinite(y).all()
    assert y.shape[0] == B
    assert y.shape[1] == 5

# --------------------------
# SeqSleepNet tests
# --------------------------


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_seqsleepnet_ctor(device):
    """Ensure SeqSleepNet constructor runs on both CPU and CUDA."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = SeqSleepNet(
        n_features=5,
        ts_len=3000,
        classes=["W", "N1", "N2", "N3", "REM"],
        sampling_rate=100,
    ).to(device)

    assert isinstance(model, SeqSleepNet)
    for p in model.parameters():
        assert p.device.type == device


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_seqsleepnet_forward(device):
    """Run one forward pass through SeqSleepNet on CPU and CUDA."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, T, D = 2, 3000, 5  # batch, time, features
    x = torch.randn(B, T, D, device=device)

    model = SeqSleepNet(
        n_features=D,
        ts_len=T,
        classes=["W", "N1", "N2", "N3", "REM"],
        sampling_rate=100,
    ).to(device)

    with torch.no_grad():
        y = model(x)

    assert torch.isfinite(y).all(), "Output contains NaNs or Infs"
    assert y.shape[0] == B, "Batch dimension mismatch"

    # Handle both (B, C) and (B, L, C) model variants
    num_classes = len(["W", "N1", "N2", "N3", "REM"])
    if y.ndim == 2:
        assert y.shape[1] == num_classes
    else:
        assert y.shape[-1] == num_classes
        assert y.shape[0] == B
        assert y.shape[1] > 0, "Sequence length should be > 0"