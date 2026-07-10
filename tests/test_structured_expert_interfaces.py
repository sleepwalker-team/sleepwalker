import torch
import pytest

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.models.MultiModel import MetaModelEntry
from sleepwalker.models.StructuredExpertInterfaces import ExpertInterfaceEdge, StructuredExpertInterfaceMetaModel


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

    def input_spec(self):
        return (1, 2, self.n_channels), {"layout": "BTC", "ts_len": 2, "n_channels": self.n_channels}


def _task_config():
    return {
        "sleep": {"task": "sleep", "labels": ["wake", "nrem"], "n_steps": 1, "target_resolution": "30s"},
        "arousal": {"task": "arousal", "labels": ["no", "yes"], "n_steps": 3, "target_resolution": "10s"},
    }


def test_structured_interfaces_slice_channels_and_emit_task_heads():
    m1 = DummyEmbeddingModel(2, 2)
    m2 = DummyEmbeddingModel(1, 3)
    model = StructuredExpertInterfaceMetaModel(
        task_config=_task_config(),
        input_channels=["a", "b", "c"],
        models=[
            MetaModelEntry(m1, ["a", "c"]),
            MetaModelEntry(m2, ["b"]),
        ],
        edges=[ExpertInterfaceEdge(source=0, target=1, bottleneck_dim=1)],
    )

    x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    y = model(x)

    assert set(y.keys()) == {"sleep", "arousal"}
    assert y["sleep"].shape == (1, 1, 2)
    assert y["arousal"].shape == (1, 3, 2)
    assert torch.equal(m1.last_input, x[:, :, [0, 2]])
    assert torch.equal(m2.last_input, x[:, :, [1]])


def test_structured_interfaces_reject_invalid_edges():
    with pytest.raises(ValueError, match="source index"):
        StructuredExpertInterfaceMetaModel(
            task_config=_task_config(),
            input_channels=["a", "b"],
            models=[
                MetaModelEntry(DummyEmbeddingModel(1, 2), ["a"]),
                MetaModelEntry(DummyEmbeddingModel(1, 2), ["b"]),
            ],
            edges=[ExpertInterfaceEdge(source=3, target=1, bottleneck_dim=1)],
        )

    with pytest.raises(ValueError, match="positive"):
        StructuredExpertInterfaceMetaModel(
            task_config=_task_config(),
            input_channels=["a", "b"],
            models=[
                MetaModelEntry(DummyEmbeddingModel(1, 2), ["a"]),
                MetaModelEntry(DummyEmbeddingModel(1, 2), ["b"]),
            ],
            edges=[ExpertInterfaceEdge(source=0, target=1, bottleneck_dim=0)],
        )


def test_zeroed_interfaces_match_no_edge_model_when_heads_are_shared():
    torch.manual_seed(7)
    m1 = DummyEmbeddingModel(1, 2)
    m2 = DummyEmbeddingModel(1, 3)
    no_edge = StructuredExpertInterfaceMetaModel(
        task_config=_task_config(),
        input_channels=["a", "b"],
        models=[
            MetaModelEntry(m1, ["a"]),
            MetaModelEntry(m2, ["b"]),
        ],
        edges=[],
    )

    with_edge = StructuredExpertInterfaceMetaModel(
        task_config=_task_config(),
        input_channels=["a", "b"],
        models=[
            MetaModelEntry(DummyEmbeddingModel(1, 2), ["a"]),
            MetaModelEntry(DummyEmbeddingModel(1, 3), ["b"]),
        ],
        edges=[ExpertInterfaceEdge(source=0, target=1, bottleneck_dim=2)],
    )
    with_edge.model_entries[0]["model"].load_state_dict(no_edge.model_entries[0]["model"].state_dict())
    with_edge.model_entries[1]["model"].load_state_dict(no_edge.model_entries[1]["model"].state_dict())
    with_edge.heads.load_state_dict(no_edge.heads.state_dict())

    for module in with_edge.edge_modules:
        torch.nn.init.zeros_(module.down.weight)
        torch.nn.init.zeros_(module.down.bias)
        torch.nn.init.zeros_(module.up.weight)
        torch.nn.init.zeros_(module.up.bias)

    x = torch.randn(4, 2, 2)
    y_no_edge = no_edge(x)
    y_with_edge = with_edge(x)

    assert torch.allclose(y_with_edge["sleep"], y_no_edge["sleep"])
    assert torch.allclose(y_with_edge["arousal"], y_no_edge["arousal"])
