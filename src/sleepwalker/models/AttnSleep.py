from __future__ import annotations

import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel


class SELayer(nn.Module):
    """Squeeze-and-excitation block for 1D feature maps."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    """Residual 1D block with squeeze-and-excitation."""
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class MRCNN(nn.Module):
    """Multi-resolution convolutional front-end used by AttnSleep."""
    def __init__(self, in_channels, afr_reduced_cnn_size):
        super().__init__()
        drate = 0.5
        self.features1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=50, stride=6, padding=24, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
        )
        self.features2 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=400, stride=50, padding=200, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks):
        downsample = nn.Sequential(nn.Conv1d(self.inplanes, planes, kernel_size=1, bias=False), nn.BatchNorm1d(planes))
        layers = [block(self.inplanes, planes, downsample=downsample)]
        layers += [block(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x = torch.cat((x1, x2), dim=2)
        x = self.dropout(x)
        return self.AFR(x)


class MultiHeadedAttention(nn.Module):
    """Standard multi-head self-attention block."""
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class EncoderLayer(nn.Module):
    """Transformer-style encoder layer used in the temporal encoder."""
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(x2))
        return x


class TCE(nn.Module):
    """Stack of temporal-context encoder layers."""
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.linears[0].in_features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward layer."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class AttnSleep(BaseModel, EmbeddingModel, ClassifierModel):
    """
    Paper: An Attention-Based Deep Learning Approach for Sleep Stage Classification With Single-Channel EEG by Eldele et al. in IEEE TRANSACTIONS ON NEURAL SYSTEMS AND REHABILITATION ENGINEERING 2021
    Code: https://github.com/emadeldeen24/AttnSleep

    Model size: ~0.5 Mio parameters which matches the original implementation

    Expected performance (sleep-edf-20 / sleep-edf-78) in the paper:
    - Accuracy:     84.4 / 81.3
    - Macro F1:     0.781 / 0.751
    - Cohens Kappa: 0.79 / 0.75

    Notes on the paper: 
      - No early stopping
      - No lr_scheduler
    """
    def __init__(
        self,
        *,
        n_channels: int,
        ts_len: int,
        classes: List[str] | None = None,
        N: int = 2,
        d_ff: int = 80,
        h: int = 5,
        dropout: float = 0.1,
        afr_reduced_cnn_size: int = 30,
        preprocessors=None,
        sequence_len: int = 1,
    ) -> None:
        """Construct the AttnSleep architecture.

        Args:
            n_channels: Number of input channels.
            ts_len: Input sequence length in samples.
            classes: Optional output class names.
            N: Number of temporal encoder layers.
            d_ff: Feed-forward hidden size in the temporal encoder.
            h: Number of attention heads.
            dropout: Dropout probability.
            afr_reduced_cnn_size: Output width of the CNN reduction block.
            preprocessors: Optional externally supplied preprocessors.
        """
        super().__init__(preprocessors=preprocessors)
        self.ts_len = ts_len
        self.n_channels = n_channels
        self.mrcnn = MRCNN(n_channels, afr_reduced_cnn_size)
        self.h = h
        self.classes = list(classes) if classes is not None else None
        self.sequence_len = int(sequence_len)
        if self.sequence_len < 1:
            raise ValueError("sequence_len must be at least 1.")
        
        with torch.no_grad():
            x = torch.zeros(1, n_channels, ts_len)
            x_feat = self.mrcnn(x)
            if x_feat.shape[2] % h != 0:
                diff = h - (x_feat.shape[2] % h)
                x_feat = F.pad(x_feat, (0, diff))
            d_model = x_feat.shape[2]

        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoder_layer = EncoderLayer(d_model, attn, ff, dropout)
        self.tce = TCE(encoder_layer, N)

        with torch.no_grad():
            x_encoded = self.tce(x_feat)
            flatten_len = x_encoded.flatten(1).shape[1]
        self._feature_dim = flatten_len
        self.fc = nn.Linear(flatten_len, self.sequence_len * len(self.classes)) if self.classes is not None else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute AttnSleep features from one input batch."""
        x = x.transpose(1, 2)
        x_feat = self.mrcnn(x)
        if x_feat.shape[2] % self.h != 0:
            diff = self.h - (x_feat.shape[2] % self.h)
            x_feat = F.pad(x_feat, (0, diff))
        x_encoded = self.tce(x_feat)
        return x_encoded.flatten(1)

    def feature_dim(self) -> int:
        """Return the dimensionality of the produced feature vector."""
        return self._feature_dim

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, int | str]]:
        """Describe the raw BTC input shape expected by ``forward``."""
        return (
            (1, self.ts_len, self.n_channels),
            {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels},
        )

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed inputs and map them to class logits."""
        x = self.encode(x)
        if self.fc is None or self.classes is None:
            raise ValueError("AttnSleep classification requires classes to be set.")
        return self.fc(x).view(x.shape[0], self.sequence_len, len(self.classes))
