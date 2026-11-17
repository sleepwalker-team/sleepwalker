"""OSF-Base encoder adapted from OSF-Open-Sleep-FM's vit1d_cls.py.

Source revision: d7e4edbc77f2b72713402234036c72b98b9b83ca.
Copyright (c) 2026 Health Intelligence Lab @ UCLA (https://github.com/yang-ai-lab).
SPDX-License-Identifier: MIT. See LICENSES/OSF.txt for the upstream notice.
Changes: retain the released CLS encoder, store blocks in ModuleList, use BTC
inputs, and add verified checkpoint loading and Sleepwalker input metadata.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import torch
from torch import nn

from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel
from sleepwalker.models.utils import resolve_checkpoint


SOURCE = {
    "repository": "https://github.com/yang-ai-lab/OSF-Open-Sleep-FM",
    "revision": "d7e4edbc77f2b72713402234036c72b98b9b83ca",
    "checkpoint_url": "https://huggingface.co/yang-ai-lab/OSF-Base/resolve/f2067f2d95b3bff43e4a690a471ba3d4457f12a7/osf_backbone.pth",
    "checkpoint_sha256": "c51190b1942556969af3c3d63c2e59430ddb1ea0377c50ea87df83712fc31857",
    "license": "MIT",
}


class OSFAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 12
        self.scale = 64 ** -0.5
        self.to_qkv = nn.Linear(768, 2304)
        self.to_out = nn.Sequential(nn.Linear(768, 768), nn.Dropout(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        q, k, v = [part.reshape(batch, tokens, self.heads, 64).transpose(1, 2) for part in self.to_qkv(x).chunk(3, dim=-1)]
        attention = ((q @ k.transpose(-1, -2)) * self.scale).softmax(dim=-1)
        return self.to_out((attention @ v).transpose(1, 2).reshape(batch, tokens, width))


class OSFPreNorm(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class OSFFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(768, 3072), nn.GELU(), nn.Dropout(0.0), nn.Linear(3072, 768), nn.Dropout(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OSFBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = OSFPreNorm(OSFAttention())
        self.ff = OSFPreNorm(OSFFeedForward())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.ff(x)


class OSFBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_patch_embedding = nn.Conv2d(1, 768, kernel_size=(4, 64), stride=(4, 64), bias=False)
        # The released checkpoint contains this embedding, but does not use it.
        self.lead_emb = nn.Embedding(3, 768)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 91, 768))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.blocks = nn.ModuleList([OSFBlock() for index in range(12)])
        self.norm = nn.LayerNorm(768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.to_patch_embedding(x.unsqueeze(1)).flatten(start_dim=2).transpose(1, 2)
        features = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), patches], dim=1) + self.pos_embedding
        for block in self.blocks:
            features = block(features)
        return self.norm(features)[:, 0]


class OSF(BaseModel, EmbeddingModel):
    """OSF-Base DINO encoder with optional released checkpoint loading.

    Input: floating point ``[B, 1920, 12]`` signals at 64 Hz (30 seconds).
    Channel order is ECG, EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP, SN,
    EOG_E1_A2, EOG_E2_A1, EEG_C3_A2, EEG_C4_A1. SpO2 is not an input.
    Prepared inputs are dimensionless, recording-normalized signals.
    Prepare recording-wide channel z-normalization and resampling in the dataset.
    The model clips prepared values to [-6, 6] after any supplied preprocessors.

    Output: ``[B, 768]`` CLS embeddings. No classification head is loaded.
    Construction initializes random weights in training mode. Loading released
    weights selects evaluation mode. Parameters remain trainable, and packages
    need no upstream checkout or weight cache after export.

    Args:
        checkpoint: File to load or download, checked against the pinned SHA256.
        allow_download: Download missing weights to checkpoint; default False.
            With no path, use ~/.cache/sleepwalker/foundation/osf/SHA256.pt.
            With neither option supplied, initialize random weights.
        preprocessors: Optional tensor preprocessing before the built-in clipping.

    Raises:
        ValueError: Invalid checksum or checkpoint metadata.
        FileNotFoundError: Weights are unavailable without download consent.
        RuntimeError: Released state cannot be loaded into the encoder.
    """

    def __init__(self, *, preprocessors: list[nn.Module] | None = None, checkpoint: str | Path | None = None, allow_download: bool = False):
        super().__init__(preprocessors=[*([] if preprocessors is None else preprocessors), nn.Hardtanh(-6.0, 6.0)])
        self.input_channels = ["ECG", "EMG_Chin", "EMG_LLeg", "EMG_RLeg", "ABD", "THX", "NP", "SN", "EOG_E1_A2", "EOG_E2_A1", "EEG_C3_A2", "EEG_C4_A1"]
        self.n_channels = len(self.input_channels)
        self.sampling_frequency = 64
        self.ts_len = 1920
        self.embedding_dim = 768
        self.source = dict(SOURCE)
        self.checkpoint_path = resolve_checkpoint("osf", url=SOURCE["checkpoint_url"], sha256=SOURCE["checkpoint_sha256"], checkpoint=checkpoint, allow_download=allow_download) if checkpoint is not None or allow_download else None
        self.backbone = OSFBackbone()
        if self.checkpoint_path is not None:
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            expected = {"model_name": "dino", "encoder_name": "vit_base", "num_leads": 12, "patch_size_time": 64, "patch_size_ch": 4, "lead_wise": 1, "sample_rate": 64, "window_size_sec": 30, "seq_len": 1920, "width": 768, "depth": 12}
            if not isinstance(payload, dict) or set(payload) != {"state_dict", "metadata"} or payload["metadata"] != expected:
                raise ValueError("OSF checkpoint must contain state_dict and the released OSF-Base metadata.")
            state = {re.sub(r"^block(\d+)\.", r"blocks.\1.", key): value for key, value in payload["state_dict"].items()}
            self.backbone.load_state_dict(state, strict=True)
            self.eval()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS embeddings for prepared BTC input (already clipped)."""
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.ts_len, self.n_channels):
            raise ValueError(f"Expected input shaped [B, {self.ts_len}, {self.n_channels}], got {tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError("OSF requires floating point signals.")
        return self.backbone(x.transpose(1, 2).contiguous())

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings without a classification head."""
        return self.encode(x)

    def feature_dim(self) -> int:
        """Return the CLS embedding width."""
        return self.embedding_dim

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        """Return BTC shape, channel order, and sample frequency."""
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels, "input_channels": list(self.input_channels), "sampling_frequency": self.sampling_frequency}
