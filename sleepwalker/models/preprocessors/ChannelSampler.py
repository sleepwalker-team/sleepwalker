import numpy as np
import torch

from sleepwalker.models.preprocessors.Preprocessor import Preprocessor

class ChannelSampler(Preprocessor):
        
    def __init__(self, n=1, **kwargs):
        super().__init__()
        self.n = n
    
    def requires_warmup(self) -> bool:
        return False

    def update(self, data):
        ...

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        channels = np.random.randint(data.shape[-1], size=self.n)
        return data[..., channels]
        