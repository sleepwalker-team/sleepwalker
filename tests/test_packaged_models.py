import torch

from sleepwalker.datasets.normalizer import ConvertUnit
from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment import PackagedModel
from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel
from sleepwalker.models.PackagedClassifierModel import PackagedClassifierModel
from sleepwalker.models.PackagedEmbeddingModel import PackagedEmbeddingModel
from sleepwalker.models.PackagedSequenceClassifierModel import PackagedSequenceClassifierModel


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


class OrderedTokenEmbedding(BaseModel, EmbeddingModel):
    def encode(self, x):
        return torch.arange(24, device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1)

    def compute(self, x):
        return self.encode(x)

    def feature_dim(self):
        return 24

    def input_spec(self):
        return (1, 60, 1), {"layout": "BTC", "ts_len": 60, "n_channels": 1}


def make_embedding_package():
    dataset = UnlabelledDataset(
        channels=[ChannelConfig('EEG', ['EEG'], preprocessors=[ConvertUnit('uV')])],
        sample_frequency=2,
        total_input="30s",
        stride="10s",
    )
    return PackagedModel(name="embedding", model=WindowEmbedding(), dataset=dataset)


def test_dataset_sample_count_is_exact_at_256_hz():
    dataset = UnlabelledDataset(
        channels=[ChannelConfig('EEG', ['EEG'], preprocessors=[ConvertUnit('uV')])],
        sample_frequency=256,
        total_input="150s",
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


def test_packaged_sequence_classifier_pools_tokens_without_losing_positions():
    dataset = UnlabelledDataset(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=2, total_input="30s", stride="30s")
    package = PackagedModel(name="tokens", model=OrderedTokenEmbedding(), dataset=dataset)
    model = PackagedSequenceClassifierModel(package=package, classes=["a", "b"], ts_len=60, sequence_len=3, token_count=6, token_dim=2, modality_count=2)

    steps = model.encode_steps(torch.zeros(2, 60, 1))

    assert steps.shape == (2, 3, 4)
    assert steps[0].tolist() == [[1.0, 2.0, 13.0, 14.0], [5.0, 6.0, 17.0, 18.0], [9.0, 10.0, 21.0, 22.0]]
    assert model(torch.zeros(2, 60, 1)).shape == (2, 3, 2)


def test_traced_imported_embedding_package_is_self_contained(tmp_path):
    encoder = torch.jit.trace(ChannelFirstEncoder(), torch.zeros(1, 1, 60))
    model = PackagedEmbeddingModel(encoder=encoder, embedding_dim=1, ts_len=60, n_channels=1, channel_first=True)
    dataset = make_embedding_package().dataset
    path = PackagedModel(name="imported", model=model, dataset=dataset).save(tmp_path / "imported")

    loaded = PackagedModel.load(path)
    assert loaded.model.features(torch.ones(2, 60, 1)).shape == (2, 1)
