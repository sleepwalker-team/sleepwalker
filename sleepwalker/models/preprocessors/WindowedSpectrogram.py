import torch

from einops import rearrange

from sleepwalker.models.preprocessors.Preprocessor import Preprocessor

# TODO: This works slightly different to the regular Spectrogram class
#       Its probably best to refactor them into one class at some point
class WindowedSpectrogram(Preprocessor):
    def __init__(self, hop_length=50, win_length=100, token_length=300, time_agg='average'):
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        self.token_length = token_length
        self.time_agg = time_agg

        # register_buffer keeps it on the correct device automatically
        window = torch.hamming_window(self.win_length, periodic=False)
        self.register_buffer("window", window)

    def _update(self, data: torch.Tensor):
        pass

    def requires_warmup(self) -> bool:
        return False

    def _transform(self, data):
        """
        Vectorized replacement for per-channel torch.stft loop.
        Input : (B, T, D)
        Output: (B, F, T', D)
        """
        x = data
        B, T, D = x.shape

        if self.window.device != x.device:
            self.window = self.window.to(x.device)

        N = T // self.token_length
        x = x.view(B, N, self.token_length, D)

        # batch all channels: (B_eff, D, T) → (B_eff*D, T)
        x_bd_t = rearrange(x, 'B N T D -> (B N D) T')

        # one batched STFT over all signals
        spec = torch.stft(
            x_bd_t,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )  

        spec = spec.abs().clamp_min(1e-6).log1p() 

        if self.time_agg == 'average':
            spec = spec.mean(-1)
        elif self.time_agg == 'concat':
            spec = rearrange(spec, 'B F T -> B (F T)')
        else:
            raise NotImplementedError('Unknown time aggregation', self.time_agg)

        spec = rearrange(spec, '(B N D) F -> B F N D', B=B, N=N)
        return spec
