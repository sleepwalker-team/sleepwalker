import torch

from sleepwalker.models.preprocessors.Preprocesser import Preprocessor

class Normalize(Preprocessor):
        
    def __init__(self):
        super().__init__()
        self.mean = None
        self.M2 = None
        self.count = 0

    def update(self, data:torch.Tensor):
        if self.mean is None or self.M2 is None:
            self.mean = data.mean(dim=0)
            self.M2 = self.mean**2
            self.count = data.shape[0]
        else:
            self.count += len(data) 
            delta = data - self.mean 
            self.mean += delta.sum(dim=0) / self.count 
            delta2 = data - self.mean 
            self.M2 += torch.sum(delta * delta2, dim=0)

    def requires_warmup(self) -> bool:
        return True

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is not None and self.M2 is not None:
            mean, var = self.mean, self.M2/self.count
            data = (data - mean) / (var.sqrt() + 1e-6)
        return data
    