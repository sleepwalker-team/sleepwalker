"""U-Time style architectures used by the repository's event-detection scripts.

This module contains an explicitly non-final PyTorch port of U-Time-inspired
models. The implementation is used by several current ``train_*.py`` scripts
for tasks such as arousal, desaturation, limb-movement, and body-position
prediction. The file also defines local normalization and convolution blocks
used only by this architecture family.

Notes:
    The original source already notes that this is an approximate port with
    several deviations from the cited implementations. The docstrings below
    therefore describe repository behavior, not paper equivalence.
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sleepwalker.utils import logger

from sleepwalker.models.Basemodel import BaseModel

class ChannelWiseNormalization(nn.Module):
    """Normalize each channel independently across the time axis."""

    def __init__(self, num_channels, eps=1e-5):
        super(ChannelWiseNormalization, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x has shape (batch_size, num_channels, time_steps)
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        x_scaled = self.gamma.view(1, -1, 1) * x_normalized + self.beta.view(1, -1, 1)
        return x_scaled

class Conv1dLayerNorm(nn.Module):
    """Apply ``LayerNorm`` to a ``Conv1d`` tensor by permuting dimensions."""

    def __init__(self, num_channels):
        super(Conv1dLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(num_channels)
    
    def forward(self, x):
        # Permute to [batch_size, length, channels] for LayerNorm
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        # Permute back to [batch_size, channels, length]
        x = x.permute(0, 2, 1)
        return x

class DepthwiseSeparableConv1d(nn.Module):
    """Implement a depthwise-separable 1D convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()

        # Depthwise convolution: one filter per input channel
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  
            bias=bias
        )

        # Pointwise (1x1) convolution: mixes channels
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvBlock(nn.Module):
    """Stack two 1D convolutions with optional normalization and dropout."""

    def __init__(self, in_channels, out_channels, activation, norm, dropout_p, kernel_size, conv):
        super(ConvBlock, self).__init__()
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "elu":
            self.act = nn.ELU()

        if norm == "batch":
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
        elif norm == "channel":
            self.norm1 = ChannelWiseNormalization(out_channels)
            self.norm2 = ChannelWiseNormalization(out_channels)
        elif norm == "layer":
            self.norm1 = Conv1dLayerNorm(out_channels)
            self.norm2 = Conv1dLayerNorm(out_channels)
        else:
            self.norm1 = None
            self.norm2 = None

        if dropout_p is not None and 0 < dropout_p < 1.0:
            self.dropout = nn.Dropout1d(p=dropout_p)
        else:
            self.dropout = None

        # Note: For (reasonable) kernel_sizes we use padding=kernel_size//2, because then the length of the time series is exactly reconstructed.
        # For example, given a 1x5x6600 input, this recreates a 1x?x6600 output before applying the mean and FC layers.
        # This is a purely empirical observation for a few different network configurations. Different values for padding are possible (and might be beneficial, dunno tbh). 
        if conv == "depthwise":
            self.conv1 = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            self.conv2 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm1:
            x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        if self.norm2:
            x = self.norm2(x)
        x = self.act(x)

        if self.dropout:
            x = self.dropout(x)

        return x

class Encoder(nn.Module):
    """Encode a sequence while storing skip connections for the decoder."""

    def __init__(self, in_channels, n_channels, maxpool, activation, norm, dropout_p, kernel_size, conv):
        super(Encoder, self).__init__()
        layers = []
        self.pools = []

        for i in range(len(n_channels)-1):
            if conv == "depthwise-first" and i == 0:
                layers.append(ConvBlock(n_channels[i],n_channels[i+1], activation, norm, dropout_p, kernel_size[i], "depthwise"))
            else:
                layers.append(ConvBlock(n_channels[i],n_channels[i+1], activation, norm, dropout_p, kernel_size[i], conv))
            self.pools.append(nn.MaxPool1d(maxpool[i]))

        self.enc_blocks = nn.ModuleList(layers)

    def forward(self, x):
        connections = []
        for block, pool in zip(self.enc_blocks, self.pools):
            x = block(x)
            connections.append(x)
            x = pool(x)
            if x.shape[-1] < 1:
                raise ValueError("Encoder output is empty after pooling. Reduce the pooling size or number of layers.")
        return x, connections

class Decoder(nn.Module):
    """Decode bottleneck features back to the input resolution."""

    def __init__(self, n_channels, upsample, activation, norm, dropout_p, kernel_size, conv):
        super(Decoder, self).__init__()
        self.upsample = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        rev_kernel_size = list(reversed(kernel_size))
        rev_n_channels = list(reversed(n_channels))
        rev_upsample = list(reversed(upsample))
        
        for i in range(len(rev_n_channels) - 1):
            self.upsample.append(torch.nn.Upsample(scale_factor=rev_upsample[i], mode="nearest"))
            
            if conv == "depthwise-first" and i == (len(rev_n_channels) - 2):
                self.dec_blocks.append(ConvBlock(2*rev_n_channels[i], rev_n_channels[i+1], activation, norm, dropout_p, rev_kernel_size[i], "depthwise"))
            else:
                self.dec_blocks.append(ConvBlock(2*rev_n_channels[i], rev_n_channels[i+1], activation, norm, dropout_p, rev_kernel_size[i], conv))

    def forward(self, x, connections):
        connections = list(reversed(connections))
        
        for upsample, dec_block, conn in zip(self.upsample, self.dec_blocks, connections):
            x = upsample(x)
            output_size = conn.size(2)
            
            if x.size(2) != output_size:
                diff = output_size - x.size(2)
                x = F.pad(x, (0, diff))  # Apply zero padding to the end of the dimension
            
            x = torch.cat([x, conn], dim=1)
            x = dec_block(x)
        
        return x

class UTime(BaseModel):
    """Build a configurable U-Time style model.

    This implementation supports both pooled classification and per-epoch
    sequence output depending on whether ``epoch_len`` is configured. Current
    training scripts use it as a flexible architecture family rather than as a
    strict reproduction of a single published configuration.

    Args:
        ts_len: Expected input length in samples.
        n_channels: Number of input channels in each window.
        sampling_frequency: Sampling interval or frequency value accepted by
            ``pandas.to_timedelta`` in current callers.
        classes: Optional class labels. When omitted, the model exposes
            features only.
        activation: Activation used inside convolution blocks.
        norm: Normalization mode for convolution blocks.
        channel: Base channel count or explicit per-layer channel sizes.
        n_layers: Number of encoder/decoder levels when ``channel`` is not
            passed as a list.
        maxpool: Pooling factor per level or a scalar repeated across levels.
        dropout_p: Dropout probability for convolution blocks.
        kernel: Kernel size per level or a scalar repeated across levels.
        mlp_size: Output feature size for pooled classification mode.
        conv: Convolution implementation variant.
        epoch_len: Optional epoch size for sequence-style outputs.
    """
    def __init__(self, 
        ts_len, 
        n_channels, 
        sampling_frequency, 
        classes=None, 
        activation = "relu", 
        norm = "channel", 
        channel = 32, 
        n_layers = None, 
        maxpool = 4, 
        dropout_p = 0.2, 
        kernel=5, 
        mlp_size = 32, 
        conv = "regular", 
        epoch_len = None
        ):
        super(UTime, self).__init__()

        self.classes = list(classes) if classes is not None else None
        self.ts_len = ts_len
        self.n_channels = n_channels
        self.fs = pd.to_timedelta(sampling_frequency)
        self.mlp_size = mlp_size

        if isinstance(maxpool, int):
            maxpool = [maxpool for _ in range(n_layers)]

        if isinstance(kernel, int):
            kernel = [kernel for _ in range(n_layers)]

        if not isinstance(channel, list) and n_layers is not None:
            channel = [channel * 2 ** i for i in range(n_layers)]

        if len(kernel) != len(maxpool):
           raise ValueError(f"Number of kernels and number of maxpool operations must be the same! {len(kernel)} !=  {len(maxpool)} (kernel_size != maxpool)") 

        channel = [self.n_channels] + channel

        self.epoch_len_str = epoch_len
        if self.epoch_len_str is not None:
            self.epoch_len_s = pd.Timedelta(epoch_len).total_seconds()
            self.samples_per_epoch = int(self.epoch_len_s / self.fs.total_seconds())
            self.avg_pool = nn.AvgPool1d(kernel_size=self.samples_per_epoch, stride=self.samples_per_epoch)
        else:
            self.epoch_len_s = None
            self.samples_per_epoch = None

        # TODO ADD dilation size 
        self.encoder = Encoder(self.n_channels, channel, maxpool, activation, norm, dropout_p, kernel, conv)
        self.bottleneck = ConvBlock(channel[-1], channel[-1], activation, norm, dropout_p, kernel[-1], "depthwise" if conv == "depthwise" else "regular")
        self.decoder = Decoder(channel, maxpool, activation, norm, dropout_p, kernel, conv)

        if self.epoch_len_s is None:
            if self.classes is not None:
                self.fc = nn.Linear(mlp_size, len(self.classes))
            else:
                self.fc = None
            self.final_conv = nn.Conv1d(channel[0], mlp_size, kernel_size=1)
            self._feature_dim = mlp_size
        else:
            if self.classes is not None:
                self.final_conv = nn.Conv1d(channel[0], len(self.classes), kernel_size=1)
            else:
                self.final_conv = nn.Conv1d(channel[0], mlp_size, kernel_size=1)
            if self.ts_len is not None and self.samples_per_epoch is not None and self.ts_len % self.samples_per_epoch == 0:
                feature_channels = len(self.classes) if self.classes is not None else mlp_size
                self._feature_dim = (self.ts_len // self.samples_per_epoch) * feature_channels
            else:
                raise ValueError("UTime requires ts_len aligned with samples_per_epoch to expose feature_dim().")

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pooled or sequence-aligned features from raw windows.

        Args:
            x: Input tensor shaped ``[batch, time, channels]``.

        Returns:
            A pooled feature tensor for classification mode or a flattened
            sequence representation when ``epoch_len`` is configured.

        Raises:
            ValueError: If the time axis cannot be segmented into epochs.
        """
        B, T, D = x.shape
        if self.samples_per_epoch is not None:
            if T % self.samples_per_epoch != 0:
                raise ValueError(f"Input time axis T={T} is not divisible by samples_per_epoch={self.samples_per_epoch}. Ensure input length matches the expected epoch segmentation.")
            N = T // self.samples_per_epoch
            x = x.view(B, N, self.samples_per_epoch, D).permute(0,1,3,2).reshape(B*N, D, self.samples_per_epoch)
        else:
            x = x.swapaxes(1,2)
        #     T = x.shape[-1]

        x, connections = self.encoder(x)
        bottleneck_embeddings = self.bottleneck(x)
        x = self.decoder(bottleneck_embeddings, connections)
        #decoded = x[:,:self.n_features,:T].swapaxes(1,2)
        x = self.final_conv(x)

        feature_embeddings = x

        if self.samples_per_epoch is not None:
            x = self.avg_pool(x)
            x = x.squeeze(-1).view(B, N * len(self.classes))
        else:
            x = x.mean(dim=2)
        return x

    def feature_dim(self) -> int:
        """Return the feature size exposed by :meth:`_features`."""
        return self._feature_dim

    def _classifier(self, x: torch.Tensor) -> torch.Tensor:
        """Project features into class logits.

        Args:
            x: Feature tensor returned by :meth:`_features`.

        Returns:
            Class logits shaped either ``[batch, classes]`` or
            ``[batch, steps, classes]`` depending on the configured output
            mode.

        Raises:
            ValueError: If the model was created without classes or if flattened
                sequence features do not align with the class dimension.
        """
        if self.classes is None:
            raise ValueError("UTime.classifier() requires classes to be set.")
        if self.samples_per_epoch is not None:
            if x.shape[-1] % len(self.classes) != 0:
                raise ValueError("Flattened UTime sequence features do not align with the class dimension.")
            return x.view(x.shape[0], x.shape[-1] // len(self.classes), len(self.classes))
        if self.fc is None:
            raise ValueError("UTime.classifier() requires classes to be set.")
        return self.fc(x)
