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

from sleepwalker.models.BaseModel import BaseModel, ClassifierModel, EmbeddingModel
from sleepwalker.models.preprocessors.Normalize import Normalize
from sleepwalker.models.preprocessors.Spectrogram import Spectrogram


class Attention(nn.Module):
    """Simple attention pooling over frame-level GRU outputs."""
    def __init__(self, input_size):
        super().__init__()
        self.att_weight = nn.Parameter(torch.randn(input_size))

    def forward(self, x):  # x: (N, T, D)
        scores = torch.tanh(x) @ self.att_weight  # (N, T)
        alpha = F.softmax(scores, dim=1)          # (N, T)
        out = torch.sum(x * alpha.unsqueeze(-1), dim=1)  # (N, D)
        return out

class SeqSleepNet(BaseModel, EmbeddingModel, ClassifierModel):
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
        n_channels=None,
        n_features=None,
        ts_len,
        classes=None,
        sampling_frequency:float | None = None,
        sampling_rate: float | None = None,
        n_fft=256,
        hop_length=100,
        epoch_len="30s",
        hidden_size = 64,
        nfilter=32,
        sequence_len=1,
        **kwargs
    ):
        """Construct the SeqSleepNet architecture.

        Args:
            n_channels: Number of input channels.
            n_features: Alias for `n_channels`.
            ts_len: Input sequence length in samples.
            classes: Optional output class names.
            sampling_frequency: Sampling frequency in Hz.
            sampling_rate: Alias for `sampling_frequency`.
            n_fft: FFT size for the spectrogram front-end.
            hop_length: Spectrogram hop length.
            epoch_len: Epoch duration used for hierarchical segmentation.
            hidden_size: Hidden size of the recurrent blocks.
            nfilter: Number of learned triangular filterbank channels.
            sequence_len: Number of center-aligned epochs to return.
        """
        self.nfilter = nfilter
        self.classes = list(classes) if classes is not None else None
        self.sampling_frequency = sampling_frequency if sampling_frequency is not None else sampling_rate
        if self.sampling_frequency is None:
            raise ValueError("SeqSleepNet requires sampling_frequency or sampling_rate.")
        n_channels = n_channels if n_channels is not None else n_features
        if n_channels is None:
            raise ValueError("SeqSleepNet requires n_channels or n_features.")
        self.epoch_len_samples = int(pd.Timedelta(epoch_len).total_seconds() * self.sampling_frequency)
        spec = [Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=int(2*self.sampling_frequency), epoch_len_samples=self.epoch_len_samples), Normalize()] 
        super().__init__(preprocessors=spec)

        self.ts_len = ts_len
        self.n_channels = n_channels
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
        self.sequence_len = int(sequence_len)
        if self.sequence_len < 1:
            raise ValueError("sequence_len must be at least 1.")
        if self.sequence_len > self.L:
            raise ValueError(f"sequence_len={self.sequence_len} exceeds the model epoch count {self.L}.")
        self._feature_dim = self.sequence_len * 2 * self.hidden_size

        if self.classes is not None:
            self.classifier_layer = nn.Linear(2 * self.hidden_size, len(self.classes))
        else:
            self.classifier_layer = None

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute hierarchical recurrent features from spectrogram inputs."""
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
        
        sequence_start = (L - self.sequence_len) // 2
        return epoch_out[:, sequence_start:sequence_start + self.sequence_len].reshape(B, -1)

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
        if self.classifier_layer is None or self.classes is None:
            raise ValueError("SeqSleepNet classification requires classes to be set.")
        feature_size = 2 * self.hidden_size
        expected = self.sequence_len * feature_size
        if x.ndim != 2 or x.shape[-1] != expected:
            raise ValueError(f"Expected SeqSleepNet features shaped [B, {expected}], got {tuple(x.shape)}.")
        x = x.view(x.shape[0], self.sequence_len, feature_size)
        return self.classifier_layer(x)
