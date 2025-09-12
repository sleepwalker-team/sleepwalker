from abc import ABC, abstractmethod
import torch

class Preprocessor(ABC):

    @abstractmethod
    def update(self, data: torch.Tensor):
        ...
    
    @abstractmethod
    def requires_warmup(self) -> bool:
        ...

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        ...
    
    