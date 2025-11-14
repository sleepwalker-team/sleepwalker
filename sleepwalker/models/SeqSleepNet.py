"""
Implementation of 

    SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging from Huy Phan, Fernando Andreotti, Navin Cooray, Oliver Y. Chén, and Maarten De Vos in IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE), vol. 27, no. 3, pp. 400-410, 2019

NOTE: The original tensorflow code + the paper were used as a basis for this implementation. Also see: https://github.com/pquochuy/SeqSleepNet/tree/master/tensorflow_net/SeqSleepNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.models.preprocessors.Normalize import Normalize
from sleepwalker.models.preprocessors.Spectrogram import Spectrogram
from sleepwalker.utils import logger


class Attention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.att_weight = nn.Parameter(torch.randn(input_size))

    def forward(self, x):  # x: (N, T, D)
        scores = torch.tanh(x) @ self.att_weight  # (N, T)
        alpha = F.softmax(scores, dim=1)          # (N, T)
        out = torch.sum(x * alpha.unsqueeze(-1), dim=1)  # (N, D)
        return out

class SeqSleepNet(BaseModel):
    """
    Paper: SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging by Phan et al. in IEEE TRANSACTIONS ON NEURAL SYSTEMS AND REHABILITATION ENGINEERING, 2019
    Code: https://github.com/pquochuy/SeqSleepNet

    Model size: ~0.1 Mio parameters. Original model size is unclear. 

    Expected performance (MASS dataset(!!)) in the paper:
    - Accuracy:     87
    - Macro F1:     0.83
    - Cohens Kappa: 0.81

    Notes on the paper:
    - The coderepository (see https://github.com/pquochuy/SeqSleepNet/blob/f8a5fd039e9511bfd5573e6cc684d8e9bfd2fab9/data_processing/preprare_raw_data.m#L144) indicates that the rawdata has been filtered by fir1 and filtfilt (Matlab) filters. The paper does not mention this filtering and we do not do it here.
    """
    def __init__(
        self,
        *,
        n_channels,
        ts_len,
        classes,
        sampling_frequency:float,  
        n_fft=256,
        hop_length=100,
        epoch_len="30s",
        hidden_size = 64,
        output_strategy = "flatten",
        nfilter=32,
        **kwargs
    ):
        self.nfilter = nfilter
        self.classes = classes
        self.sampling_frequency = sampling_frequency
        self.epoch_len_samples = int(pd.Timedelta(epoch_len).total_seconds() * sampling_frequency)
        spec = [Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=int(2*sampling_frequency), epoch_len_samples=self.epoch_len_samples), Normalize()] 
        super().__init__(preprocessors=spec)

        self.ts_len = ts_len
        T = self.ts_len
        if T % self.epoch_len_samples != 0:
            raise ValueError(f"Input length T={T} is not divisible by epoch_len={self.epoch_len_samples} samples → expected multiple of {self.epoch_len_samples} samples per epoch.")
        
        self.L = T // self.epoch_len_samples
        self.D = n_channels
        self.F = n_fft // 2 + 1
        self.n_fft = n_fft

        # Triangular filterbank matrix T
        W_bl_np = self._lin_tri_filter_shape(self.nfilter, n_fft, self.sampling_frequency)
        self.register_buffer("W_bl", torch.tensor(W_bl_np, dtype=torch.float32))  # (F, nfilter)

        # One learnable filterbank weight matrix per channel
        self.filter_weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.F, self.nfilter)) for _ in range(self.D)
        ])

        self.hidden_size = hidden_size
        self.frame_gru = nn.GRU(
            input_size=self.nfilter * self.D,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.frame_attention = Attention(2 * self.hidden_size)
        self.epoch_gru = nn.GRU(
            input_size=2 * self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.output_strategy = output_strategy

        if self.output_strategy == "flatten":
            L = self.ts_len // self.epoch_len_samples
            self.classifier = nn.Linear(2 * self.hidden_size * L, len(self.classes))
        else:
            self.classifier = nn.Linear(2 * self.hidden_size, len(self.classes))

    def _lin_tri_filter_shape(self, nfilt, nfft, samplerate, lowfreq=0, highfreq=None):
        """
        Generates a triangular filter bank matrix for feature extraction in the frequency domain.

        NOTE: Initial version is adopted from here from here https://github.com/pquochuy/SeqSleepNet/blob/master/tensorflow_net/SeqSleepNet/filterbank_shape.py
        
        Args:
            nfilt (int): Number of filters in the filter bank.
            nfft (int): Number of FFT points.
            samplerate (float): Sampling rate of the signal in Hz.
            lowfreq (float, optional): Lowest frequency in the filter bank. Defaults to 0.
            highfreq (float, optional): Highest frequency in the filter bank. Defaults to samplerate/2.

        Returns:
            np.ndarray: A 2D array of shape (nfft//2 + 1, nfilt) representing the filter bank matrix.
                        Each column corresponds to a filter, and each row corresponds to a frequency bin.
        """
        # 
        highfreq = min(highfreq or samplerate / 2, samplerate / 2)
        lowfreq = max(lowfreq, 0)
        hzpoints = np.linspace(lowfreq, highfreq, nfilt + 2)
        bin = np.clip(np.floor((nfft + 1) * hzpoints / samplerate).astype(int), 0, nfft // 2)
        hzpoints = np.linspace(lowfreq, highfreq, nfilt + 2)
        fbank = np.zeros([nfilt, nfft // 2 + 1])
        for j in range(nfilt):
            fbank[j, bin[j]:bin[j+1]] = (np.arange(bin[j], bin[j+1]) - bin[j]) / (bin[j+1] - bin[j])
            fbank[j, bin[j+1]:bin[j+2]] = (bin[j+2] - np.arange(bin[j+1], bin[j+2])) / (bin[j+2] - bin[j+1])
        return fbank.T.astype(np.float32)

    # def forward(self, x, **kwargs):
    #     # Implement a custom forward method here, because we need to reshape the spectograms
    #     B, T, D = x.shape
        
    #     x = x.view(B * self.L, self.epoch_len_samples, D)  # reshape to (B*L, epoch_len_samples, D)
    #     super()._forward(x)
    #     if self.is_warmup():
    #         if self.preprocessors:
    #             self.preprocessors.update(x)
    #         return torch.zeros( (B, self.n_classes()), device=x.device) 
    #     else:
    #         if self.preprocessors is not None:
    #             x = self.preprocessors(x)
    #         # Cannot call super().forward here, because during warmup phase the shapes do not match anymore (i.e. batch size is wrong)
    #         return self._forward(x, **kwargs)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Filterbanks ---
        D = x.shape[-1]
        B = x.shape[0] // self.L
        filtered = []
        for d in range(D):
            S = x[..., d]  # (B*L, F, T')
            Wc_fb = torch.sigmoid(self.filter_weights[d]) * self.W_bl  # (F, M)
            X = torch.matmul(Wc_fb.T, S)  # (M, T')
            X = X.permute(0, 2, 1)  # (B*L, T', M)
            filtered.append(X)

        filtered = torch.cat(filtered, dim=-1)  # (B*L, T', M*D)

        B_L, T, F = filtered.shape
        filtered = filtered.view(B, self.L, T, F)  # (B, L, T', M*D)

        # --- Frame-level GRU ---
        B, L, T_frame, feat_dim = filtered.shape
        frame_input = filtered.view(B * L, T_frame, feat_dim)  # (B*L, T', M*D)

        frame_out, _ = self.frame_gru(frame_input)  # (B*L, T', 2*H)
        attended = self.frame_attention(frame_out)  # (B*L, 2*H)
        epoch_reps = attended.view(B, L, -1)  # (B, L, 2*H)

        # --- Epoch-level GRU ---
        epoch_out, _ = self.epoch_gru(epoch_reps)  # (B, L, 2*H)
        
        if self.output_strategy == "flatten":
            epoch_out = epoch_out.contiguous()
            epoch_out = epoch_out.view(B, -1)  # (B, 2*L*H)
        elif self.output_strategy == "window-right":
            epoch_out = epoch_out[:, -1, :]  # (B, 2*H)
        elif self.output_strategy == "window-middle":
            epoch_out = epoch_out[:, L // 2, :]  # (B, 2*H)
        elif self.output_strategy == "sequence":
            epoch_out = epoch_out.reshape(B*L, -1)
        else:
            valid_strategies = ["flatten", "window-right", "window-middle", "sequence"]
            raise ValueError(f"Unknown output strategy: {self.output_strategy}. Valid strategies are: {valid_strategies}")

        logits = self.classifier(epoch_out)

        if self.output_strategy == "sequence":
            logits = logits.view(B, L, len(self.classes))

        return logits
