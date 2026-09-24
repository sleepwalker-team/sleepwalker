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
        center: Whether to center-pad before the STFT. When false, frames are
            defined by ``win_length`` exactly, matching MATLAB ``spectrogram``.
        drop_dc: Remove the zero-frequency bin.
        preserve_epochs: Keep the epoch axis instead of folding it into batch.
        log_scale: ``"log1p"`` or decibel magnitude ``"db"``.
    """

    def __init__(self, n_fft=256, hop_length=64, win_length=None, epoch_len_samples=None, center=True, drop_dc=False, preserve_epochs=False, log_scale="log1p", channels=None):
        super().__init__(channels=channels)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.epoch_len_samples = epoch_len_samples
        self.center = bool(center)
        self.drop_dc = bool(drop_dc)
        self.preserve_epochs = bool(preserve_epochs)
        self.log_scale = log_scale
        if self.preserve_epochs and self.epoch_len_samples is None:
            raise ValueError("preserve_epochs requires epoch_len_samples.")
        if self.log_scale not in {"log1p", "db"}:
            raise ValueError("log_scale must be 'log1p' or 'db'.")

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

        n_epochs = 1
        if self.epoch_len_samples is not None:
            if T % self.epoch_len_samples != 0:
                raise ValueError(f"Input length {T} is not divisible by epoch_len_samples={self.epoch_len_samples}.")
            L = T // self.epoch_len_samples
            n_epochs = L
            x = x.view(B * L, self.epoch_len_samples, D)

        # batch all channels: (B_eff, D, T) → (B_eff*D, T)
        B_eff = x.shape[0]
        x_bd_t = x.permute(0, 2, 1).reshape(B_eff * D, -1)

        # one batched STFT over all signals
        if self.center:
            spec = torch.stft(
                x_bd_t,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                return_complex=True,
            )
        else:
            if x_bd_t.shape[-1] < self.win_length:
                raise ValueError(f"Input length {x_bd_t.shape[-1]} is shorter than win_length={self.win_length}.")
            frames = x_bd_t.unfold(-1, self.win_length, self.hop_length)
            frames = frames * self.window
            spec = torch.fft.rfft(frames, n=self.n_fft, dim=-1).transpose(1, 2)

        magnitude = spec.abs().clamp_min(1e-6)
        spec = magnitude.log1p() if self.log_scale == "log1p" else 20 * torch.log10(magnitude)
        if self.drop_dc:
            spec = spec[:, 1:]
        spec = spec.view(B_eff, D, spec.size(1), spec.size(2))
        spec = spec.permute(0, 2, 3, 1).contiguous()  # (B_eff, F, T', D)
        if self.preserve_epochs:
            spec = spec.view(B, n_epochs, spec.shape[1], spec.shape[2], spec.shape[3])
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
    
