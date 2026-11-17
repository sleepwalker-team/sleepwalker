import pytest
import torch

from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel
from sleepwalker.models.MultiTaskClassifierModel import MultiTaskClassifierModel


class MeanEncoder(BaseModel, EmbeddingModel):
    def encode(self, data):
        return data.mean(dim=1)

    def feature_dim(self):
        return 3

    def input_spec(self):
        return (1, 10, 3), {"layout": "BTC", "ts_len": 10, "n_channels": 3}


def test_multitask_classifier_shapes():
    model = MultiTaskClassifierModel(encoder=MeanEncoder(), outputs={"sleep": {"classes": ["wake", "sleep"], "sequence_len": 1}, "event": {"classes": ["no", "yes"], "sequence_len": 4}})

    outputs = model(torch.randn(5, 10, 3))

    assert outputs["sleep"].shape == (5, 1, 2)
    assert outputs["event"].shape == (5, 4, 2)


def test_multitask_classifier_rejects_non_embedding_encoder():
    with pytest.raises(TypeError, match="EmbeddingModel"):
        MultiTaskClassifierModel(encoder=torch.nn.Identity(), outputs={"task": {"classes": ["a"], "sequence_len": 1}})
