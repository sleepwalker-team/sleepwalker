import torch

from sleepwalker.models.preprocessors.Preprocessor import Preprocessor

class ZNormalize(Preprocessor):
        
    def __init__(self, use_global_statistics=False, **kwargs):
        super().__init__()
        self.use_global_statistics = use_global_statistics
        self.count = 0

    def update(self, x:torch.Tensor):
        """
        Add a new mini‑batch `x` with shape (..., m).
        Works with e.g. (batch, time, features) = (n, k, m).

        Algorithm: parallelised Welford (Chan et al., 1979).
        """
        if x.numel() == 0:
            return                                    # nothing to do

        # Flatten all but last dim -> (num_obs, m)
        obs      = x.reshape(-1, x.shape[-1])
        k        = obs.size(0)                        # new observations
        batch_mu = obs.mean(dim=0)
        batch_M2 = ((obs - batch_mu).pow(2)).sum(dim=0)

        if self.count == 0:
            self.count = k
            self.mean = batch_mu
            self.M2 = batch_M2
            return 

        # Merge batch stats with running stats
        delta       = batch_mu - self.mean
        total_n     = self.count + k

        self.mean  += delta * k / total_n
        self.M2    += batch_M2 + delta.pow(2) * self.count * k / total_n
        self.count  = total_n

    def requires_warmup(self) -> bool:
        return False

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # data shape: (batch_size, T_s, D)
        n_channels = data.shape[2]
        if self.use_global_statistics and self.count > 0:
            mean = self.mean.reshape(1, 1, n_channels)
            std = torch.sqrt(self.M2 / self.count).reshape(1, 1, n_channels)
        else:
            # Estimate from current window
            mean = data.mean(dim=1, keepdim=True)
            std = data.std(dim=1, keepdim=True)
        return (data - mean) / (std + 1e-8)  