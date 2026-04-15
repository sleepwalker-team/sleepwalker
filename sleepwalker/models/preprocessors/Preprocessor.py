"""Abstract base class for tensor preprocessors.

Preprocessors are small modules attached to models and warmed externally by
trainer code when needed. Current tests and model code assume tensors shaped
`[B, T, C]`.
"""

from abc import ABC, abstractmethod
import torch
from torch import nn 

class Preprocessor(nn.Module, ABC):
    """Common interface for repository preprocessors.

    Subclasses must define whether they need warmup, how warmup updates are
    accumulated, and how they transform one input tensor at inference/training
    time.
    """

    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    @abstractmethod
    def update(self, data: torch.Tensor):
        """Update internal running statistics from a warmup batch."""
        ...
    
    @abstractmethod
    def requires_warmup(self) -> bool:
        """Return whether this preprocessor needs warmup before use."""
        ...

    @torch.inference_mode()
    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Transform one batch tensor."""
        ...
    
    
