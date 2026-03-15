import torch

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.models.MetaModel import MetaModel, MetaModelEntry


class DummyEmbeddingModel(BaseModel):
    def __init__(self, n_channels, classes):
        super().__init__()
        self.n_channels = n_channels
        self.classes = classes
        self.proj = torch.nn.Linear(n_channels, len(classes), bias=False)
        self.last_input = None

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return self.proj(x.mean(dim=1))


def test_metamodel_slices_channels_and_projects_embeddings():
    m1 = DummyEmbeddingModel(2, ["e1", "e2"])
    m2 = DummyEmbeddingModel(1, ["e3"])

    model = MetaModel(
        classes=["c1", "c2", "c3"],
        input_channels=["a", "b", "c"],
        models=[
            MetaModelEntry("m1", m1, ["a", "c"]),
            MetaModelEntry("m2", m2, ["b"]),
        ],
    )

    x = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    )
    y = model(x)

    assert y.shape == (1, 3)
    assert torch.equal(m1.last_input, x[:, :, [0, 2]])
    assert torch.equal(m2.last_input, x[:, :, [1]])
