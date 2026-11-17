import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import torch
import torch.nn as nn


def resolve_checkpoint(model_name: str, *, url: str, sha256: str, checkpoint: str | Path | None = None, allow_download: bool = False) -> Path:
    """Resolve and verify a released checkpoint, with downloads disabled by default.

    Args:
        model_name: Cache subdirectory for the model.
        url: Pinned released checkpoint URL.
        sha256: Expected released checkpoint checksum.
        checkpoint: File to load or download. Defaults to
            ``~/.cache/sleepwalker/foundation/MODEL/SHA256.pt``.
        allow_download: Allow a missing checkpoint to be downloaded to that file.

    Returns:
        Verified checkpoint path.

    Raises:
        ValueError: Checksum mismatch.
        FileNotFoundError: Missing file without download consent.
        OSError: Download or filesystem failure. Partial files are removed.
    """
    path = Path(checkpoint).expanduser() if checkpoint is not None else Path.home() / ".cache" / "sleepwalker" / "foundation" / model_name / f"{sha256}.pt"
    temporary = None
    try:
        if not path.exists():
            if not allow_download:
                raise FileNotFoundError(f"Missing {model_name} checkpoint: {path}. Set allow_download=True to download the released weights.")
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        candidate = path if temporary is None else temporary
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != sha256:
            raise ValueError(f"Checkpoint {candidate} SHA256 mismatch: expected {sha256}, got {actual}.")
        if temporary is not None:
            os.replace(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        num_timescales = (dim + 1) // 2
        div_term = torch.exp(torch.arange(0, num_timescales).float() * (-math.log(10000.0) / dim))
        angle_rates = position * div_term

        pe[:, 0::2] = torch.sin(angle_rates[:, : dim // 2 + dim % 2])
        pe[:, 1::2] = torch.cos(angle_rates[:, : dim // 2])

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)

class AttentionPooling(nn.Module):
    def __init__(self, dim, attn_size, along_dimension=1):
        super().__init__()
        self.Wa = nn.Linear(dim, attn_size)
        self.ae = nn.Parameter(torch.randn(attn_size))
        self.ba = nn.Parameter(torch.zeros(attn_size))
        self.along_dimension = along_dimension

    def forward(self, x):
        a_t = torch.tanh(self.Wa(x) + self.ba)
        e_t = torch.matmul(a_t, self.ae)
        alpha = torch.softmax(e_t, dim=self.along_dimension)
        out = torch.sum(x * alpha.unsqueeze(-1), dim=self.along_dimension)
        return out, alpha
