from abc import ABC, abstractmethod
import torch
from torch import nn 

class Preprocessor(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self, data: torch.Tensor):
        ...
    
    @abstractmethod
    def requires_warmup(self) -> bool:
        ...

    @torch.inference_mode()
    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        ...
    
    