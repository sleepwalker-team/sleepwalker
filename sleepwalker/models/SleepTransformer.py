from __future__ import annotations

import math
import pandas as pd
import torch
import torch.nn as nn

from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel
from sleepwalker.models.preprocessors.Normalize import Normalize
from sleepwalker.models.preprocessors.Spectrogram import Spectrogram

class SinusoidalPositionalEncoding(nn.Module):
    """Add sinusoidal positional encodings to a sequence tensor."""
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
    """Attention-weighted pooling over the sequence dimension."""
    def __init__(self, dim, attn_size):
        super().__init__()
        self.Wa = nn.Linear(dim, attn_size)
        self.ae = nn.Parameter(torch.randn(attn_size))
        self.ba = nn.Parameter(torch.zeros(attn_size))

    def forward(self, x):
        a_t = torch.tanh(self.Wa(x) + self.ba)
        e_t = torch.matmul(a_t, self.ae)
        alpha = torch.softmax(e_t, dim=1)
        out = torch.sum(x * alpha.unsqueeze(-1), dim=1)
        return out, alpha

class TransformerBlock(nn.Module):
    """Transformer encoder block with optional attention pooling."""
    def __init__(
        self,
        input_dim,
        max_len,
        num_layers,
        num_heads,
        d_ff,
        dropout,
        use_attention_pooling=False,
        attn_size=None,
    ):
        super().__init__()

        self.use_attention_pooling = use_attention_pooling
        self.pos_enc = SinusoidalPositionalEncoding(input_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if self.use_attention_pooling:
            if attn_size is None:
                raise ValueError("attn_size must be set when use_attention_pooling=True")
            self.attn_pool = AttentionPooling(input_dim, attn_size)

    def forward(self, x):
        x = self.pos_enc(x)
        x = self.transformer(x)
        if self.use_attention_pooling:
            x, alpha = self.attn_pool(x)
            return x, alpha
        else:
            return x

class SleepTransformer(BaseModel, EmbeddingModel, ClassifierModel):
    """
    Paper: SleepTransformer: Automatic Sleep Staging With Interpretability and Uncertainty Quantification by Phan et al. in IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, 2022
    Code: https://github.com/pquochuy/SleepTransformer

    Model size: 3.7 Mio (paper) vs. 3.8 Mio parameters (this config)

    Expected performance (sleep-edf-78) in the paper:
    - Accuracy:     84.9
    - Macro F1:     0.788
    - Cohens Kappa: 0.789

    Notes on the paper: 
      - No lr_scheduler used
      - Early stopping in the paper validates every 100 training steps (batches maybe?) and not after each epoch.
      - The coderepository (https://github.com/pquochuy/SleepTransformer/blob/26167f26ddd8bb59ce787e56655c6e99ba446d2d/shhs/process_and_save_1file.m#L28) indicates that the rawdata has been filtered by fir1 and filtfilt (Matlab) filters. The paper does not mention this filtering. 
    """
    def __init__(
        self,
        *,
        ts_len=None,
        sampling_frequency=100,
        classes=None,         # class names (C)
        n_channels,           # input channels (from dataset)
        ndim=128,             # spectral bins (F)
        frame_seq_len=29,     # T (frames per epoch)
        epoch_seq_len=21,     # L (epochs per input)
        epoch_len="30s",
        frame_window="2s",
        frame_hop="1s",
        frm_d_ff=1024,
        frm_num_blocks=4,
        frm_num_heads=8,
        frm_attention_dropout=0.1,
        frm_attention_size=64,
        seq_d_ff=1024,
        seq_num_blocks=4,
        seq_num_heads=8,
        seq_attention_dropout=0.1,
        fc_hidden_size=1024,
        fc_dropout=0.1,
        sequence_len=1,
    ):
        """Construct the SleepTransformer architecture.

        Args:
            classes: Optional output class names. When omitted, the classifier
                head is disabled and the model acts as a feature extractor.
            n_channels: Number of input channels.
            ndim: Number of spectral bins after the spectrogram front-end.
            frame_seq_len: Number of frames per epoch for the first transformer.
            epoch_seq_len: Number of epochs per input window for the second
                transformer.
            sampling_frequency: Raw signal sampling frequency in Hz.
            epoch_len: Duration of one sleep-staging epoch.
            frame_window: Spectrogram frame duration within each epoch.
            frame_hop: Spectrogram frame step within each epoch.
            frm_d_ff: Feed-forward size in the frame transformer.
            frm_num_blocks: Number of frame-transformer layers.
            frm_num_heads: Attention heads for the frame transformer.
            frm_attention_dropout: Dropout in the frame transformer.
            frm_attention_size: Attention-pooling hidden size.
            seq_d_ff: Feed-forward size in the sequence transformer.
            seq_num_blocks: Number of sequence-transformer layers.
            seq_num_heads: Attention heads for the sequence transformer.
            seq_attention_dropout: Dropout in the sequence transformer.
            fc_hidden_size: Hidden size of the classifier MLP.
            fc_dropout: Dropout in the classifier MLP.
            sequence_len: Number of center-aligned epochs to return.
        """
        self.sampling_frequency = float(sampling_frequency)
        self.epoch_len_samples = int(round(pd.to_timedelta(epoch_len).total_seconds() * self.sampling_frequency))
        frame_window_samples = int(round(pd.to_timedelta(frame_window).total_seconds() * self.sampling_frequency))
        frame_hop_samples = int(round(pd.to_timedelta(frame_hop).total_seconds() * self.sampling_frequency))
        expected_frame_seq_len = 1 + (self.epoch_len_samples - frame_window_samples) // frame_hop_samples
        if expected_frame_seq_len != frame_seq_len:
            raise ValueError(f"frame_seq_len={frame_seq_len} does not match epoch_len={epoch_len}, frame_window={frame_window}, and frame_hop={frame_hop}; expected {expected_frame_seq_len} frames.")
        spec = [
            Spectrogram(
                n_fft=2 * ndim,
                hop_length=frame_hop_samples,
                win_length=frame_window_samples,
                epoch_len_samples=self.epoch_len_samples,
                center=False,
                drop_dc=True,
                preserve_epochs=True,
                log_scale="db",
            ),
            Normalize(stat_dims=[2, 4]),
        ]
        super().__init__(preprocessors=spec)
        self.classes = list(classes) if classes is not None else None

        self.ndim = ndim
        self.ts_len = ts_len
        self.n_channels = n_channels
        self.nchannel = n_channels
        self.nclass = len(self.classes) if self.classes is not None else 0
        self.frame_seq_len = frame_seq_len
        self.epoch_seq_len = epoch_seq_len
        self.sequence_len = int(sequence_len)
        if self.sequence_len < 1:
            raise ValueError("sequence_len must be at least 1.")
        if self.sequence_len > self.epoch_seq_len:
            raise ValueError(f"sequence_len={self.sequence_len} exceeds epoch_seq_len={self.epoch_seq_len}.")
        expected_ts_len = self.epoch_seq_len * self.epoch_len_samples
        if self.ts_len is not None and self.ts_len != expected_ts_len:
            raise ValueError(f"SleepTransformer expects epoch_seq_len * epoch_len = {expected_ts_len} samples, got ts_len={self.ts_len}.")

        self.stft_n_fft = 2 * self.ndim
        self.stft_hop_length = frame_hop_samples
        self.frm_input_dim = self.ndim * self.nchannel

        if self.frm_input_dim % frm_num_heads != 0:
            new_frm_num_heads = 0
            for i in range(frm_num_heads, 0, -1):
                if self.frm_input_dim % i == 0:
                    new_frm_num_heads = i
                    break
            frm_num_heads = new_frm_num_heads

        self.epoch_transformer = TransformerBlock(
            input_dim=self.frm_input_dim,
            max_len=frame_seq_len,
            num_layers=frm_num_blocks,
            num_heads=frm_num_heads,
            d_ff=frm_d_ff,
            dropout=frm_attention_dropout,
            use_attention_pooling=True,
            attn_size=frm_attention_size,
        )

        self.sequence_transformer = TransformerBlock(
            input_dim=self.frm_input_dim,
            max_len=epoch_seq_len,
            num_layers=seq_num_blocks,
            num_heads=seq_num_heads,
            d_ff=seq_d_ff,
            dropout=seq_attention_dropout,
            use_attention_pooling=False,
        )

        self._feature_dim = self.sequence_len * self.frm_input_dim
        if self.classes is not None:
            self.fc = nn.Sequential(
                nn.Linear(self.frm_input_dim, fc_hidden_size),
                nn.ReLU(),
                nn.Dropout(fc_dropout),
                nn.Linear(fc_hidden_size, fc_hidden_size),
                nn.ReLU(),
                nn.Dropout(fc_dropout),
                nn.Linear(fc_hidden_size, len(self.classes)),
            )
        else:
            self.fc = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SleepTransformer features from spectrogram inputs."""
        if x.ndim != 5:
            raise ValueError(f"Expected spectrograms shaped [B, L, F, T, C], got {tuple(x.shape)}.")
        B, L, F, T, D = x.shape
        expected_shape = (self.epoch_seq_len, self.ndim, self.frame_seq_len, self.n_channels)
        if (L, F, T, D) != expected_shape:
            raise ValueError(f"Expected spectrogram dimensions {expected_shape} ([L, F, T, C]), got {(L, F, T, D)}.")
        x_ep = x.permute(0, 1, 3, 2, 4).reshape(B * L, T, F * D)
        x_ep, _ = self.epoch_transformer(x_ep)
        x_ep = x_ep.view(B, L, -1)
        x_seq = self.sequence_transformer(x_ep)

        sequence_start = (self.epoch_seq_len - self.sequence_len) // 2
        return x_seq[:, sequence_start:sequence_start + self.sequence_len].reshape(B, -1)

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
            raise ValueError("SleepTransformer classification requires classes to be set.")
        expected = self.sequence_len * self.frm_input_dim
        if x.ndim != 2 or x.shape[-1] != expected:
            raise ValueError(f"Expected SleepTransformer features shaped [B, {expected}], got {tuple(x.shape)}.")
        x = x.view(x.shape[0], self.sequence_len, self.frm_input_dim)
        return self.fc(x)
        
