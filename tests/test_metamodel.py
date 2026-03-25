import torch
from torch.utils.data import DataLoader

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.models.MetaModel import MetaModel, MetaModelEntry


class DummyEmbeddingModel(BaseModel):
    def __init__(self, n_channels, feature_dim):
        super().__init__()
        self.n_channels = n_channels
        self._feature_dim = feature_dim
        self.proj = torch.nn.Linear(n_channels, feature_dim, bias=False)
        self.last_input = None

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return self.proj(x.mean(dim=1))

    def feature_dim(self) -> int:
        return self._feature_dim

    def _classifier(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AddConstant(torch.nn.Module):
    def __init__(self, value: float, warmup: bool = False):
        super().__init__()
        self.value = value
        self.warmup = warmup
        self.update_calls = 0

    def requires_warmup(self):
        return self.warmup

    def update(self, x: torch.Tensor):
        self.update_calls += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.value


def test_metamodel_slices_channels_fuses_embeddings_and_applies_task_heads():
    m1 = DummyEmbeddingModel(2, 2)
    m2 = DummyEmbeddingModel(1, 1)

    model = MetaModel(
        task_config={
            "sleep staging": {"task": "sleep staging", "labels": ["c1", "c2", "c3"], "n_steps": 1, "target_resolution": "30s"},
            "breathing": {"task": "breathing", "labels": ["b1", "b2"], "n_steps": 3, "target_resolution": "10s"},
        },
        input_channels=["a", "b", "c"],
        models=[
            MetaModelEntry(m1, ["a", "c"]),
            MetaModelEntry(m2, ["b"]),
        ],
    )

    x = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    )
    y = model(x)

    assert set(y.keys()) == {"sleep staging", "breathing"}
    assert y["sleep staging"].shape == (1, 1, 3)
    assert y["breathing"].shape == (1, 3, 2)
    assert torch.equal(m1.last_input, x[:, :, [0, 2]])
    assert torch.equal(m2.last_input, x[:, :, [1]])


def test_metamodel_applies_and_warms_nested_preprocessors():
    meta_pre = AddConstant(1.0, warmup=True)
    sub_pre = AddConstant(2.0, warmup=True)
    m1 = DummyEmbeddingModel(1, 1)
    m1.preprocessors.append(sub_pre)

    model = MetaModel(
        task_config={
            "sleep staging": {"task": "sleep staging", "labels": ["c1"], "n_steps": 1, "target_resolution": "30s"},
        },
        input_channels=["a"],
        models=[MetaModelEntry(m1, ["a"])],
        preprocessors=[meta_pre],
    )

    x = torch.zeros(1, 2, 1)
    _ = model(x)

    assert torch.equal(m1.last_input, torch.full((1, 2, 1), 3.0))

    loader = DataLoader([{"data": x[0], "target": torch.zeros(1, 1, dtype=torch.long)}], batch_size=1)
    model._warmup_preprocessors(loader, device="cpu")

    assert meta_pre.update_calls == 1
    assert sub_pre.update_calls == 1
