import torch

from sleepwalker.models.preprocessors.Preprocessor import Preprocessor

class Spectogram(Preprocessor):
    def __init__(self, n_fft = 256, hop_length = 64, win_length = None, epoch_len_samples = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.epoch_len_samples = epoch_len_samples
    
        if self.win_length is None:
            self.window = torch.hamming_window(self.n_fft)
        else:
            self.window = torch.hamming_window(self.win_length)

    def update(self, data: torch.Tensor):
        ...

    def requires_warmup(self) -> bool:
        return False

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        x = data
        B, T, D = x.shape
        if self.window.device != x.device:
            self.window = self.window.to(x.device)
        
        if self.epoch_len_samples is not None:
            L = T // self.epoch_len_samples
            x = x.view(B * L, self.epoch_len_samples, D)

        specs = [
            torch.stft(x[:, :, d], n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, win_length=self.win_length, return_complex=True).abs().clamp(min=1e-6).log1p()
            for d in range(D)
        ]

        specs = torch.stack(specs, dim=-1)
        # specs has shape (B, F, T', D)
        # B:  Batch size
        # F:  Frequencies
        # T': Length of time series
        # D:  Number of channels
        return specs
    