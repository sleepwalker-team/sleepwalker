"""U-Sleep architecture with continuous encoder/decoder sequence processing."""

import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel
from sleepwalker.models.preprocessors.RobustScaler import RobustScaler


class USleep(BaseModel, EmbeddingModel, ClassifierModel):
    """Implement the published U-Sleep encoder, decoder, and sequence head.

    The complete input window is processed continuously. The dense temporal
    representation is center-aligned to ``sequence_len`` epochs, average
    pooled per epoch, and mapped to logits shaped ``[B, S, C]``.
    """

    def __init__(
        self,
        *,
        ts_len,
        n_channels,
        classes=None,
        sampling_frequency,
        depth=12,
        init_filters=5,
        kernel_size=9,
        dilation=1,
        epoch_len="30s",
        activation="elu",
        dense_classifier_activation="tanh",
        transition_window=1,
        complexity_factor=2,
        sequence_len=1,
    ):
        super().__init__(preprocessors=[RobustScaler()])

        self.classes = list(classes) if classes is not None else None
        self.ts_len = int(ts_len)
        self.n_channels = int(n_channels)
        self.sampling_frequency = float(sampling_frequency)
        self.depth = int(depth)
        self.sequence_len = int(sequence_len)
        if self.depth < 1:
            raise ValueError("depth must be at least 1.")
        if self.sequence_len < 1:
            raise ValueError("sequence_len must be at least 1.")
        if kernel_size % 2 == 0:
            raise ValueError("USleep requires an odd kernel_size for same padding.")
        if activation not in {"elu", "relu", "tanh"}:
            raise ValueError(f"Unknown activation: {activation}")
        if dense_classifier_activation not in {"elu", "relu", "tanh"}:
            raise ValueError(f"Unknown dense classifier activation: {dense_classifier_activation}")
        if complexity_factor <= 0:
            raise ValueError("complexity_factor must be positive.")

        self.activation_name = activation
        self.dense_activation_name = dense_classifier_activation
        self.epoch_len_str = epoch_len
        self.epoch_len_s = pd.to_timedelta(epoch_len).total_seconds()
        samples_per_epoch = self.epoch_len_s * self.sampling_frequency
        rounded_samples = round(samples_per_epoch)
        if samples_per_epoch <= 0 or not math.isclose(samples_per_epoch, rounded_samples, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"epoch_len={epoch_len} does not contain an integer number of samples at {sampling_frequency} Hz.")
        self.samples_per_epoch = int(rounded_samples)
        output_samples = self.sequence_len * self.samples_per_epoch
        if output_samples > self.ts_len:
            raise ValueError(f"sequence_len={self.sequence_len} and epoch_len={epoch_len} require {output_samples} samples, but ts_len={self.ts_len}.")

        channel_factor = math.sqrt(float(complexity_factor))
        base_filters = int(init_filters)
        in_channels = self.n_channels
        self.encoder_convs = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        encoder_channels = []
        for _ in range(self.depth):
            out_channels = int(base_filters * channel_factor)
            self.encoder_convs.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, dilation=dilation))
            self.encoder_norms.append(nn.BatchNorm1d(out_channels))
            encoder_channels.append(out_channels)
            in_channels = out_channels
            base_filters = int(base_filters * math.sqrt(2))

        bottom_channels = int(base_filters * channel_factor)
        self.bottom_conv = nn.Conv1d(in_channels, bottom_channels, kernel_size, padding=kernel_size // 2)
        self.bottom_norm = nn.BatchNorm1d(bottom_channels)

        self.upsample_convs = nn.ModuleList()
        self.upsample_norms = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        in_channels = bottom_channels
        for residual_channels in reversed(encoder_channels):
            base_filters = int(math.ceil(base_filters / math.sqrt(2)))
            out_channels = int(base_filters * channel_factor)
            self.upsample_convs.append(nn.Conv1d(in_channels, out_channels, 2, padding="same"))
            self.upsample_norms.append(nn.BatchNorm1d(out_channels))
            self.decoder_convs.append(nn.Conv1d(out_channels + residual_channels, out_channels, kernel_size, padding=kernel_size // 2))
            self.decoder_norms.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        dense_base_channels = len(self.classes) if self.classes is not None else in_channels
        self.dense_channels = max(1, int(dense_base_channels * channel_factor))
        self.dense_conv = nn.Conv1d(in_channels, self.dense_channels, 1)
        self._feature_dim = self.sequence_len * self.dense_channels
        if self.classes is not None:
            self.sequence_conv1 = nn.Conv1d(self.dense_channels, len(self.classes), transition_window, padding="same")
            self.sequence_conv2 = nn.Conv1d(len(self.classes), len(self.classes), transition_window, padding="same")
        else:
            self.sequence_conv1 = None
            self.sequence_conv2 = None

    def activate(self, x, activation_name):
        if activation_name == "elu":
            return F.elu(x)
        if activation_name == "tanh":
            return torch.tanh(x)
        return F.relu(x)

    def match_residual(self, x, residual):
        """Center-crop or pad decoder features to a residual connection."""
        difference = x.shape[-1] - residual.shape[-1]
        if difference > 0:
            start = difference // 2 + difference % 2
            return x[..., start:start + residual.shape[-1]]
        if difference < 0:
            missing = -difference
            return F.pad(x, (missing // 2, missing // 2 + missing % 2))
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.n_channels:
            raise ValueError(f"Expected input shaped [B, T, {self.n_channels}], got {tuple(x.shape)}.")
        x = x.transpose(1, 2)
        residuals = []
        for conv, norm in zip(self.encoder_convs, self.encoder_norms):
            x = norm(self.activate(conv(x), self.activation_name))
            x = F.pad(x, (x.shape[-1] % 2, 0))
            residuals.append(x)
            x = F.max_pool1d(x, 2)

        x = self.bottom_norm(self.activate(self.bottom_conv(x), self.activation_name))
        for up_conv, up_norm, decoder_conv, decoder_norm, residual in zip(self.upsample_convs, self.upsample_norms, self.decoder_convs, self.decoder_norms, reversed(residuals)):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = up_norm(self.activate(up_conv(x), self.activation_name))
            x = self.match_residual(x, residual)
            x = torch.cat([residual, x], dim=1)
            x = decoder_norm(self.activate(decoder_conv(x), self.activation_name))

        x = self.activate(self.dense_conv(x), self.dense_activation_name)
        output_samples = self.sequence_len * self.samples_per_epoch
        if output_samples > x.shape[-1]:
            raise ValueError(f"USleep needs {output_samples} decoded samples for sequence output, but produced {x.shape[-1]}.")
        start = (x.shape[-1] - output_samples) // 2
        x = x[..., start:start + output_samples]
        x = F.avg_pool1d(x, self.samples_per_epoch, self.samples_per_epoch)
        return x.transpose(1, 2).reshape(x.shape[0], -1)

    def feature_dim(self) -> int:
        return self._feature_dim

    def input_spec(self) -> tuple[tuple[int, ...], dict[str, int | str]]:
        return (
            (1, self.ts_len, self.n_channels),
            {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels},
        )

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        if self.classes is None or self.sequence_conv1 is None or self.sequence_conv2 is None:
            raise ValueError("USleep classification requires classes to be set.")
        expected = self.sequence_len * self.dense_channels
        if x.ndim != 2 or x.shape[-1] != expected:
            raise ValueError(f"Expected USleep features shaped [B, {expected}], got {tuple(x.shape)}.")
        x = x.view(x.shape[0], self.sequence_len, self.dense_channels).transpose(1, 2)
        x = self.activate(self.sequence_conv1(x), self.activation_name)
        return self.sequence_conv2(x).transpose(1, 2)
