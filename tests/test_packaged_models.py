import torch

from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment import PackagedModel
from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel
from sleepwalker.models.PackagedClassifierModel import PackagedClassifierModel
from sleepwalker.models.PackagedEmbeddingModel import PackagedEmbeddingModel
from tools.foundation_models import SleepFMClinicalEncoder, trace_encoder


class WindowEmbedding(BaseModel, EmbeddingModel):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 3)

    def encode(self, x):
        return self.projection(x.mean(dim=1))

    def compute(self, x):
        return self.encode(x)

    def feature_dim(self):
        return 3

    def input_spec(self):
        return (1, 60, 1), {"layout": "BTC", "ts_len": 60, "n_channels": 1}


class ChannelFirstEncoder(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=-1)


class ParameterizedEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 3)

    def forward(self, x):
        return self.projection(x)


class MaskAwareBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, x, mask):
        contextual_embeddings = (x + mask.to(dtype=x.dtype).unsqueeze(-1)) * self.scale
        return contextual_embeddings.mean(dim=1), contextual_embeddings


def make_embedding_package():
    dataset = UnlabelledDataset(
        channels=[ChannelConfig("EEG", ["EEG"], unit="uV")],
        sample_frequency=2,
        total_input="30s",
        target_resolution="30s",
        stride="10s",
    )
    return PackagedModel(name="embedding", model=WindowEmbedding(), dataset=dataset)


def test_dataset_sample_count_is_exact_at_256_hz():
    dataset = UnlabelledDataset(
        channels=[ChannelConfig("EEG", ["EEG"], unit="uV")],
        sample_frequency=256,
        total_input="150s",
        target_resolution="30s",
    )

    assert dataset.get_timeseries_len() == 38400


def test_packaged_classifier_model_stacks_native_windows_and_trains_a_head(tmp_path):
    package_path = make_embedding_package().save(tmp_path / "embedding")
    model = PackagedClassifierModel(
        package=package_path,
        classes=["negative", "positive"],
        ts_len=120,
        sequence_len=2,
        freeze_encoder=True,
    )

    x = torch.randn(4, 120, 1)
    assert model.n_embedding_windows == 4
    assert model.feature_dim() == 12
    assert model.features(x).shape == (4, 12)
    assert model(x).shape == (4, 2, 2)
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    model.train()
    assert model.training
    assert not model.encoder.training


def test_packaged_classifier_model_rejects_non_exact_outer_windows():
    package = make_embedding_package()
    try:
        PackagedClassifierModel(package=package, classes=["negative", "positive"], ts_len=111)
    except ValueError as error:
        assert "exact number" in str(error)
    else:
        raise AssertionError("Expected invalid outer geometry to fail.")


def test_traced_imported_embedding_package_is_self_contained(tmp_path):
    encoder = torch.jit.trace(ChannelFirstEncoder(), torch.zeros(1, 1, 60))
    model = PackagedEmbeddingModel(encoder=encoder, embedding_dim=1, ts_len=60, n_channels=1, channel_first=True)
    dataset = make_embedding_package().dataset
    path = PackagedModel(name="imported", model=model, dataset=dataset).save(tmp_path / "imported")

    loaded = PackagedModel.load(path)
    assert loaded.model.features(torch.ones(2, 60, 1)).shape == (2, 1)


def test_foundation_trace_keeps_state_as_movable_parameters():
    traced, embedding_dim = trace_encoder(ParameterizedEncoder(), torch.zeros(1, 2))

    assert embedding_dim == 3
    assert list(traced.parameters())
    traced.to(dtype=torch.float64)
    assert all(parameter.dtype == torch.float64 for parameter in traced.parameters())
    assert traced(torch.zeros(1, 2, dtype=torch.float64)).dtype == torch.float64


def test_sleepfm_trace_creates_padding_masks_on_the_input_device():
    encoder = SleepFMClinicalEncoder(MaskAwareBackbone(), [[0], [1], [2], [3]])
    traced, embedding_dim = trace_encoder(encoder, torch.zeros(1, 4, 2))

    assert embedding_dim == 8
    traced.to(device="meta")
    assert traced(torch.zeros(2, 4, 2, device="meta")).shape == (2, 8)
