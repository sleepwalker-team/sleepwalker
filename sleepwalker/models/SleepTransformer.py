from __future__ import annotations

import math
import torch
import torch.nn as nn

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.models.preprocessors.Normalize import Normalize
from sleepwalker.models.preprocessors.Spectrogram import Spectrogram

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

class SleepTransformer(BaseModel):
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
        classes,              # class names (C)
        n_channels,           # input channels (from dataset)
        ndim=128,             # spectral bins (F)
        frame_seq_len=29,     # T (frames per epoch)
        epoch_seq_len=21,     # L (epochs per input)
        hop_length=64,
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
        output_strategy="center",
    ):
        spec = [Spectrogram(n_fft=2 * (ndim - 1), hop_length=hop_length), Normalize()] 
        super().__init__(preprocessors=spec)
        self.classes = list(classes)

        self.ndim = ndim
        self.nchannel = n_channels
        self.nclass = len(classes)
        self.frame_seq_len = frame_seq_len
        self.epoch_seq_len = epoch_seq_len
        self.output_strategy = output_strategy

        self.stft_n_fft = 2 * (self.ndim - 1)
        self.stft_hop_length = hop_length
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

        classifier_input_dim = self.frm_input_dim * epoch_seq_len if output_strategy == "flatten" else self.frm_input_dim
        self.fc = nn.Sequential(
            nn.Linear(classifier_input_dim, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size, len(classes)),
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        spec = x.permute(0, 2, 1, 3).reshape(B, x.shape[2], -1)
        total_frames = spec.shape[1] // self.frame_seq_len
        usable_T = total_frames * self.frame_seq_len
        if total_frames < self.epoch_seq_len:
            raise ValueError("Input too short to form the required number of epochs.")
        spec = spec[:, :usable_T, :].view(B, total_frames, self.frame_seq_len, -1)[:, : self.epoch_seq_len]

        x_ep = spec.reshape(-1, self.frame_seq_len, spec.shape[-1])
        x_ep, _ = self.epoch_transformer(x_ep)
        x_ep = x_ep.view(B, self.epoch_seq_len, -1)
        x_seq = self.sequence_transformer(x_ep)

        if self.output_strategy == "center":
            idx = self.epoch_seq_len // 2
            x_out = x_seq[:, idx, :]
        elif self.output_strategy == "last":
            x_out = x_seq[:, -1, :]
        elif self.output_strategy == "flatten":
            x_out = x_seq.reshape(B, -1)
        elif self.output_strategy == "sequence":
            x_out = x_seq.reshape(B * self.epoch_seq_len, -1)
        else:
            raise ValueError("Unknown output_strategy")

        yhat = self.fc(x_out)
        if self.output_strategy == "sequence":
            yhat = yhat.reshape(B, self.epoch_seq_len, -1)
        return yhat
