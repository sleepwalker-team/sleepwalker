import numpy as np
import pandas as pd
import pytest
import torch

from sleepwalker.models.DRIVEN import DRIVEN, EfficientNetV2Branch, SamePadConv2d, prepare_target
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.trainer.utils.targets import prepare_multitask_target


def test_driven_reference_heads_embeddings_and_independent_parameters():
    model = DRIVEN()
    model.eval()
    x = torch.randn(1, 3520, 2)
    x[..., 1] = 95
    with torch.no_grad():
        logits = model(x)
        embedded = model.features(x)
    assert list(logits) == ["abd", "spo2_0", "spo2_10", "spo2_15", "spo2_20", "spo2_25"]
    assert all(value.shape == (1, 1, 2) for value in logits.values())
    expected = torch.cat([value[:, 0].softmax(-1) for value in logits.values()], dim=-1)
    torch.testing.assert_close(embedded, expected)
    assert embedded.shape == (1, model.feature_dim()) == (1, 12)
    assert model.input_spec()[0] == (1, 3520, 2)
    parameters = [{id(parameter) for parameter in branch.parameters()} for branch in model.branches.values()]
    assert sum(len(group) for group in parameters) == len(set.union(*parameters))
    target = pd.DataFrame({"sleep": 1, "respiratory_event": 0, "hypopnea_unknown": 0}, index=pd.date_range("2025-01-01", periods=3520, freq="15.625ms"))
    target.loc[target.index[:64], "respiratory_event"] = 1
    item = prepare_multitask_target(target, task_config=model.task_config)
    assert item["target"].shape == (6, 1, 2)
    assert torch.equal(item["target"][:, 0], torch.tensor([[0., 1.]]).expand(6, 2))
    trainer = MultiLabelTrainer(epochs=1, optimizer=lambda model: torch.optim.Adam(model.parameters()), task_config=model.task_config, device="cpu")
    assert list(trainer.classification_contract()["tasks"]) == list(logits)
    model.feature_variant = "pooled"
    with torch.no_grad():
        assert model.features(x).shape == (1, 7680)
        processed = model.apply_preprocessors(x)
        model.preprocessors = torch.nn.ModuleList([])
        model.ts_len, model.n_channels = 1920, 6
        model.feature_variant = "probabilities"
        torch.testing.assert_close(model.features(processed), embedded)
        assert model.input_spec()[0] == (1, 1920, 6)


def test_branch_singleton_width_same_padding_and_gradient():
    branch = EfficientNetV2Branch()
    branch.train()
    data = torch.randn(2, 64, requires_grad=True)
    logits, pooled = branch(data)
    assert pooled.shape == (2, 1280)
    logits.square().mean().backward()
    assert data.grad is not None and torch.isfinite(data.grad).all()
    conv = SamePadConv2d(1, 1, 3, stride=2)
    assert conv(torch.ones(1, 1, 64, 1)).shape == (1, 1, 32, 1)


def test_driven_target_checks_only_supervised_span_and_each_annotator():
    config = {name: {"labels": ["regular", "event"], "default": "regular", "target_resolution": "30s", "target_offset": "0s", "sequence_len": 1, "percentage": 1 / 30} for name in ["first", "second"]}
    target = pd.DataFrame({"sleep": np.ones(55), "event": np.zeros(55), "hypopnea_unknown": np.zeros(55)}, index=pd.date_range("2025-01-01", periods=55, freq="1s"))
    target.loc[target.index[45], "hypopnea_unknown"] = 1
    assert prepare_target(target, task_config=config)["target_mask"].all()
    alternate = target.copy()
    alternate.iloc[0, 2] = 1
    item = prepare_target(target, alternate, task_config=config)
    assert "target_extra" not in item
    target.iloc[0, 2] = 1
    assert prepare_target(target, task_config=config) is None
    target.iloc[0, 2] = 0
    target.iloc[0, 0] = 0
    assert prepare_target(target, task_config=config) is None
