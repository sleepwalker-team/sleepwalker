import torch

from sleepwalker.models.preprocessors.Preprocessor import Preprocessor

class NormalizeAlongDim(Preprocessor):
    def __init__(self, dim = 0):
        super().__init__()
        self.dim = dim

    def update(self, data:torch.Tensor):
        ...

    def requires_warmup(self) -> bool:
        return False 

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return (data - data.mean(dim=self.dim, keepdim=True)) / (data.var(dim=self.dim, keepdim=True).sqrt() + 1e-6)
    