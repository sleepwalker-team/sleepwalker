"""Short-time Fourier transform preprocessor."""

import torch
import torch.nn.functional as F

from sleepwalker.models.preprocessors.Preprocessor import Preprocessor

class Spectrogram(Preprocessor):
    """Convert time-domain inputs into log-magnitude spectrograms.

    Args:
        n_fft: FFT size passed to `torch.stft`.
        hop_length: Hop length between analysis windows.
        win_length: Window length used for the Hamming window. Defaults to
            `n_fft`.
        epoch_len_samples: Optional epoch length that splits one input sequence
            into several shorter spectrogram examples.
    """

    def __init__(self, n_fft=256, hop_length=64, win_length=None, epoch_len_samples=None, channels=None):
        super().__init__(channels=channels)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.epoch_len_samples = epoch_len_samples

        # register_buffer keeps it on the correct device automatically
        window = torch.hamming_window(self.win_length, periodic=False)
        self.register_buffer("window", window)
 
    def _update(self, data: torch.Tensor):
        """No-op warmup hook because spectrogram extraction is stateless."""
        pass

    def requires_warmup(self) -> bool:
        """Return whether this preprocessor requires warmup."""
        return False

    @torch.inference_mode()
    def _transform(self, data: torch.Tensor) -> torch.Tensor:
        """Convert batched signals into batched spectrogram tensors.

        Args:
            data: Input tensor shaped `(B, T, D)`.

        Returns:
            A tensor shaped `(B_eff, F, T', D)`, where `B_eff` may be larger
            than `B` when `epoch_len_samples` is used.
        """
        x = data
        B, T, D = x.shape

        if self.window.device != x.device:
            self.window = self.window.to(x.device)

        if self.epoch_len_samples is not None:
            L = T // self.epoch_len_samples
            x = x.view(B * L, self.epoch_len_samples, D)

        # batch all channels: (B_eff, D, T) → (B_eff*D, T)
        B_eff = x.shape[0]
        x_bd_t = x.permute(0, 2, 1).reshape(B_eff * D, -1)

        # one batched STFT over all signals
        spec = torch.stft(
            x_bd_t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length or self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )  # (B_eff*D, F, T')

        spec = spec.abs().clamp_min(1e-6).log1p()     # (B_eff*D, F, T')
        spec = spec.view(B_eff, D, spec.size(1), spec.size(2))
        spec = spec.permute(0, 2, 3, 1).contiguous()  # (B_eff, F, T', D)
        return spec

# class Spectrogram(Preprocessor):
#     def __init__(self, n_fft = 256, hop_length = 64, win_length = None, epoch_len_samples = None):
#         super().__init__()
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.epoch_len_samples = epoch_len_samples
    
#         if self.win_length is None:
#             self.window = torch.hamming_window(self.n_fft)
#         else:
#             self.window = torch.hamming_window(self.win_length)

#     def update(self, data: torch.Tensor):
#         ...

#     def requires_warmup(self) -> bool:
#         return False

#     def __call__(self, data: torch.Tensor) -> torch.Tensor:
#         print(data.shape)
#         x = data
#         B, T, D = x.shape
#         if self.window.device != x.device:
#             self.window = self.window.to(x.device)
        
#         if self.epoch_len_samples is not None:
#             L = T // self.epoch_len_samples
#             x = x.view(B * L, self.epoch_len_samples, D)

#         specs = [
#             torch.stft(x[:, :, d], n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, win_length=self.win_length, return_complex=True).abs().clamp(min=1e-6).log1p()
#             for d in range(D)
#         ]

#         specs = torch.stack(specs, dim=-1)
#         # specs has shape (B, F, T', D)
#         # B:  Batch size
#         # F:  Frequencies
#         # T': Length of time series
#         # D:  Number of channels
#         print(specs.shape)
#         return specs
    
