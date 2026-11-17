"""SleepFM clinical encoder adapted to Sleepwalker's native model interface.

Adapted from zou-group/sleepfm-clinical, revision
2bcbae04c3592f61352addb7ac3d4193f0a3ca25, sleepfm/models/models.py.
Copyright (c) 2025 Rahul Thapa. SPDX-License-Identifier: CC-BY-NC-4.0
License: https://creativecommons.org/licenses/by-nc/4.0/
Changes: retain the released base encoder, use BTC inputs, and add verified
checkpoint loading, framework metadata, and embedding output for four modalities.
See LICENSES/SleepFM.txt for the upstream notice.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel
from sleepwalker.models.utils import resolve_checkpoint


SOURCE = {
    "repository": "https://github.com/zou-group/sleepfm-clinical",
    "revision": "2bcbae04c3592f61352addb7ac3d4193f0a3ca25",
    "checkpoint_url": "https://raw.githubusercontent.com/zou-group/sleepfm-clinical/2bcbae04c3592f61352addb7ac3d4193f0a3ca25/sleepfm/checkpoints/model_base/best.pt",
    "checkpoint_sha256": "ffc9fc10233ebc4d0aae71abce87070db51b8bac6e4b16a5c1b4401a5f73f799",
    "license": "CC-BY-NC-4.0",
}


class SleepFMTokenizer(nn.Module):
    """Encode each five-second channel patch with the released CNN."""

    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 1
        for index, out_channels in enumerate([4, 8, 16, 32, 64, 128]):
            layers.extend([nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(out_channels), nn.ELU(), nn.LayerNorm([out_channels, 640 // (2 ** (index + 1))])])
            in_channels = out_channels
        layers.extend([nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 128)])
        self.tokenizer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, samples = x.shape
        patches = x.reshape(batch * channels * (samples // 640), 1, 640)
        return self.tokenizer(patches).reshape(batch, channels, samples // 640, 128)


class SleepFMAttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dropout=0.0, batch_first=True)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # The released encoder bypasses spatial attention for a single channel.
        if key_padding_mask is not None and key_padding_mask.shape[1] == 1:
            return x.mean(dim=1)
        return self.transformer_layer(x, src_key_padding_mask=key_padding_mask).mean(dim=1)


class SleepFMPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        position = torch.arange(128).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 128, 2) * (-math.log(10000.0) / 128))
        encoding = torch.zeros(128, 128)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1]]


class SleepFMBackbone(nn.Module):
    """Released SetTransformer architecture with checkpoint-compatible names."""

    def __init__(self):
        super().__init__()
        self.patch_embedding = SleepFMTokenizer()
        self.spatial_pooling = SleepFMAttentionPooling()
        self.positional_encoding = SleepFMPositionalEncoding()
        self.layer_norm = nn.LayerNorm(128)
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dropout=0.0, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=6, enable_nested_tensor=False)
        self.temporal_pooling = SleepFMAttentionPooling()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(x)
        batch, channels, tokens, width = patches.shape
        spatial = patches.permute(0, 2, 1, 3).reshape(batch * tokens, channels, width)
        mask = torch.zeros(spatial.shape[:2], dtype=torch.bool, device=x.device)
        features = self.spatial_pooling(spatial, mask).reshape(batch, tokens, width)
        return self.transformer_encoder(self.layer_norm(self.positional_encoding(features)))


class SleepFM(BaseModel, EmbeddingModel):
    """SleepFM clinical base encoder with optional released checkpoint loading.

    Input: floating point ``[B, ts_len, 13]`` recording-normalized signals at
    128 Hz. Default ``ts_len=38400`` is five minutes. Channel order is ECG,
    EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP, SpO2, SN, EOG_E1_A2,
    EOG_E2_A1, EEG_C3_A2, EEG_C4_A1. Prepared inputs are dimensionless;
    prepare recording-wide channel z-normalization and resampling in the dataset.

    Output: ``[B, 4 * (ts_len // 640) * 128]`` contextual embeddings, flattened
    in BAS, RESP, ECG, EMG order, then time, then feature. The default has 30720
    features. No classification head is loaded. Parameters remain trainable.
    Construction initializes random weights in training mode. Loading released
    weights selects evaluation mode. No upstream checkout is needed.

    Args:
        checkpoint: File to load or download, checked against the pinned SHA256.
        allow_download: Download missing weights to checkpoint; default False.
            With no path, use ~/.cache/sleepwalker/foundation/sleepfm/SHA256.pt.
            With neither option supplied, initialize random weights.
        ts_len: Positive multiple of 640, at most 81920 (128 five-second tokens).
        preprocessors: Optional tensor preprocessing, as for other BaseModel classes.

    Raises:
        ValueError: Invalid window, checksum, or checkpoint payload.
        FileNotFoundError: Weights are unavailable without download consent.
        RuntimeError: Released state cannot be loaded into the encoder.

    SleepFM's adapted encoder and released weights carry a CC BY-NC 4.0 license.
    """

    def __init__(self, *, ts_len: int = 38400, preprocessors: list[nn.Module] | None = None, checkpoint: str | Path | None = None, allow_download: bool = False):
        super().__init__(preprocessors=preprocessors)
        if not isinstance(ts_len, int) or isinstance(ts_len, bool) or ts_len < 640 or ts_len > 81920 or ts_len % 640:
            raise ValueError("SleepFM ts_len must be a positive multiple of 640, at most 81920.")
        self.input_channels = ["ECG", "EMG_Chin", "EMG_LLeg", "EMG_RLeg", "ABD", "THX", "NP", "SpO2", "SN", "EOG_E1_A2", "EOG_E2_A1", "EEG_C3_A2", "EEG_C4_A1"]
        self.n_channels = len(self.input_channels)
        self.sampling_frequency = 128
        self.ts_len = ts_len
        self.embedding_dim = 4 * (ts_len // 640) * 128
        self.source = dict(SOURCE)
        self.checkpoint_path = resolve_checkpoint("sleepfm", url=SOURCE["checkpoint_url"], sha256=SOURCE["checkpoint_sha256"], checkpoint=checkpoint, allow_download=allow_download) if checkpoint is not None or allow_download else None
        self.backbone = SleepFMBackbone()
        self.modality_channel_indices = ((9, 10, 11, 12), (4, 5, 6, 7, 8), (0,), (1, 2, 3))
        if self.checkpoint_path is not None:
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or set(payload) != {"state_dict"}:
                raise ValueError("SleepFM checkpoint must contain only state_dict.")
            self.backbone.load_state_dict({key.removeprefix("module."): value for key, value in payload["state_dict"].items()}, strict=True)
            self.eval()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return contextual modality embeddings for prepared BTC signals."""
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.ts_len, self.n_channels):
            raise ValueError(f"Expected input shaped [B, {self.ts_len}, {self.n_channels}], got {tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError("SleepFM requires floating point signals.")
        return torch.cat([self.backbone(x[:, :, indices].transpose(1, 2).contiguous()).flatten(start_dim=1) for indices in self.modality_channel_indices], dim=1)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings without a classification head."""
        return self.encode(x)

    def feature_dim(self) -> int:
        """Return the number of contextual embedding features per window."""
        return self.embedding_dim

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        """Return BTC shape, channel order, and sample frequency."""
        return (1, self.ts_len, self.n_channels), {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels, "input_channels": list(self.input_channels), "sampling_frequency": self.sampling_frequency}
