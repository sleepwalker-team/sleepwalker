import torch
from sleepwalker.models.preprocessors.Preprocessor import Preprocessor

class Normalize(Preprocessor):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", None)
        self.register_buffer("M2", None)
        self.count = 0
        # self.register_buffer("count", torch.tensor(0.0))

    def update(self, data: torch.Tensor):
        # Batch statistics
        batch_count = data.shape[0]
        batch_mean = data.mean(dim=0)
        batch_M2 = ((data - batch_mean) ** 2).sum(dim=0)

        if self.mean is None:
            # Initialize with first batch
            self.mean = batch_mean
            self.M2 = batch_M2
            self.count = batch_count
        else:
            delta = batch_mean - self.mean
            total_count = self.count + batch_count

            # Update running mean and M2
            new_mean = self.mean + delta * batch_count / total_count
            new_M2 = self.M2 + batch_M2 + delta**2 * self.count * batch_count / total_count

            # Commit updates
            self.mean = new_mean
            self.M2 = new_M2
            self.count = total_count

    def requires_warmup(self) -> bool:
        return True

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is not None and self.M2 is not None and self.count > 1:
            var = self.M2 / (self.count - 1)
            data = (data - self.mean) / (var.sqrt() + 1e-6)
        return data

    