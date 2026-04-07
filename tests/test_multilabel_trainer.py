import pandas as pd
import pytest
import torch
from functools import partial
from torch.utils.data import DataLoader, Dataset

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.models.MetaModel import MetaModel, MetaModelEntry
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.trainer.losses import class_weights_for_loss, estimate_multilabel_class_cnts


class DummyEmbeddingModel(BaseModel):
    def __init__(self, n_channels, feature_dim):
        super().__init__()
        self.proj = torch.nn.Linear(n_channels, feature_dim, bias=False)
        self._feature_dim = feature_dim

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=1))

    def feature_dim(self) -> int:
        return self._feature_dim

    def _classifier(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MultiLabelBatchDataset(Dataset):
    target_resolution = "30s"

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_trainer(**kwargs):
    task_config = {
        "sleep staging": {
            "labels": ["wake", "n1", "n2", "n3", "rem"],
            "default": None,
            "percentage": 0.5,
            "target_resolution": "30s",
        },
        "breathing": {
            "labels": ["apnea", "hypopnea", "regular breathing"],
            "default": "regular breathing",
            "percentage": 0.5,
            "target_resolution": "10s",
        },
    }
    task_config = MultiLabelTrainer.normalize_task_config(task_config)
    task_config = {
        task: {**cfg, "loss_function": torch.nn.functional.cross_entropy}
        for task, cfg in task_config.items()
    }
    return MultiLabelTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        task_config=task_config,
        device="cpu",
        warmup_device="cpu",
        **kwargs,
    )


def build_dataset(items):
    return MultiLabelBatchDataset(items)


def make_target_df():
    idx = pd.date_range("2024-01-01", periods=30, freq="1s")
    target = pd.DataFrame(0, index=idx, columns=["wake", "n1", "n2", "n3", "rem", "apnea", "hypopnea", "regular breathing"])
    target.loc[idx[:10], "apnea"] = 1
    target.loc[idx[10:20], "regular breathing"] = 1
    target.loc[idx[20:], "hypopnea"] = 1
    target.loc[idx[:], "n2"] = 1
    return target


def test_trainer_builds_task_metadata():
    trainer = build_trainer()

    assert trainer.classes == ["wake", "n1", "n2", "n3", "rem", "apnea", "hypopnea", "regular breathing"]
    assert trainer.task_config["sleep staging"]["n_steps"] == 1
    assert trainer.task_config["breathing"]["n_steps"] == 3
    assert trainer.max_task_steps == 3


def test_build_multitask_target_uses_task_steps():
    trainer = build_trainer()
    y = trainer.build_multitask_target(make_target_df(), trainer.task_config)

    assert y.shape == (2, 3)
    assert y[0].tolist() == [2, -1, -1]
    assert y[1].tolist() == [0, 2, 1]


def test_build_multitask_target_rejects_multiple_active_labels_within_step():
    trainer = build_trainer()
    target = make_target_df()
    target.loc[target.index[:10], "hypopnea"] = 1

    with pytest.raises(ValueError, match="Task 'breathing', step 0"):
        trainer.build_multitask_target(target, trainer.task_config)


def test_build_multitask_target_returns_none_when_raise_error_is_false():
    trainer = build_trainer()
    target = make_target_df()
    target.loc[target.index[:10], "hypopnea"] = 1

    assert trainer.build_multitask_target(target, trainer.task_config, raise_error=False) is None


def test_build_multitask_target_uses_task_specific_percentage():
    task_config = {
        "sleep staging": {"labels": ["wake", "n1"], "default": None, "percentage": 0.8, "target_resolution": "20s"},
        "breathing": {"labels": ["apnea", "regular breathing"], "default": "regular breathing", "percentage": 0.2, "target_resolution": "10s"},
    }
    task_config = MultiLabelTrainer.normalize_task_config(task_config)
    idx = pd.date_range("2024-01-01", periods=20, freq="1s")
    target = pd.DataFrame(0, index=idx, columns=["wake", "n1", "apnea", "regular breathing"])
    target.loc[idx[:17], "n1"] = 1
    target.loc[idx[:3], "apnea"] = 1
    target.loc[idx[10:12], "apnea"] = 1

    y = MultiLabelTrainer.build_multitask_target(target, task_config)

    assert y.tolist() == [[1, -1], [0, 1]]


def test_trainer_rejects_non_divisible_task_resolution():
    task_config = {
        "sleep staging": {"labels": ["wake", "n1"], "default": None, "target_resolution": "30s"},
        "breathing": {"labels": ["apnea", "regular breathing"], "default": "regular breathing", "target_resolution": "7s"},
    }

    with pytest.raises(ValueError, match="must divide"):
        MultiLabelTrainer(
            epochs=1,
            optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
            task_config={
                task: {**cfg, "loss_function": torch.nn.functional.cross_entropy}
                for task, cfg in MultiLabelTrainer.normalize_task_config(task_config).items()
            },
            device="cpu",
            warmup_device="cpu",
        )


def test_trainer_rejects_unknown_condition_label():
    with pytest.raises(ValueError, match="condition_labels contains labels"):
        build_trainer(condition_task="sleep staging", condition_labels=["apnea"], conditioned_tasks=["breathing"])


def test_run_epoch_masks_conditioned_tasks_during_wake():
    task_config = MultiLabelTrainer.normalize_task_config({
        "sleep": {"labels": ["wake", "n2"], "default": None, "target_resolution": "30s"},
        "breathing": {"labels": ["apnea", "regular"], "default": "regular", "target_resolution": "10s"},
    })
    dataset = MultiLabelBatchDataset([
        {
            "data": torch.zeros(4, 1),
            "target": torch.tensor([[1, -1, -1], [0, 0, 0]], dtype=torch.long),
        },
        {
            "data": torch.zeros(4, 1),
            "target": torch.tensor([[0, -1, -1], [1, 1, 1]], dtype=torch.long),
        },
    ])
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = MetaModel(
        task_config=task_config,
        input_channels=["sig"],
        models=[MetaModelEntry(DummyEmbeddingModel(1, 1), ["sig"])],
    )
    with torch.no_grad():
        model.model_entries[0]["model"].proj.weight.zero_()
        model.heads["sleep"].weight.zero_()
        model.heads["sleep"].bias.copy_(torch.tensor([0.0, 1.0]))
        model.heads["breathing"].weight.zero_()
        model.heads["breathing"].bias.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))

    masked_trainer = MultiLabelTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        task_config={task: {**cfg, "loss_function": torch.nn.functional.cross_entropy} for task, cfg in task_config.items()},
        condition_task="sleep",
        condition_labels=["n2"],
        conditioned_tasks=["breathing"],
        device="cpu",
        warmup_device="cpu",
    )
    unmasked_trainer = MultiLabelTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        task_config={task: {**cfg, "loss_function": torch.nn.functional.cross_entropy} for task, cfg in task_config.items()},
        device="cpu",
        warmup_device="cpu",
    )

    masked_loss, masked_cm = masked_trainer.run_epoch(loader, None, model, "TEST")
    unmasked_loss, unmasked_cm = unmasked_trainer.run_epoch(loader, None, model, "TEST")

    assert masked_cm["breathing"].tolist() == [[3, 0], [0, 0]]
    assert unmasked_cm["breathing"].tolist() == [[3, 0], [3, 0]]
    assert masked_cm["sleep"].tolist() == [[0, 1], [0, 1]]
    assert masked_loss < unmasked_loss


def test_run_epoch_condition_mask_applies_to_entire_sample_when_condition_task_has_multiple_steps():
    task_config = MultiLabelTrainer.normalize_task_config({
        "coarse": {"labels": ["off", "on"], "default": None, "target_resolution": "30s"},
        "sleep": {"labels": ["wake", "n2"], "default": None, "target_resolution": "10s"},
        "breathing": {"labels": ["apnea", "regular"], "default": "regular", "target_resolution": "10s"},
    })
    dataset = MultiLabelBatchDataset([
        {
            "data": torch.zeros(4, 1),
            "target": torch.tensor([[0, -1, -1], [0, 1, 0], [0, 1, 0]], dtype=torch.long),
        },
        {
            "data": torch.zeros(4, 1),
            "target": torch.tensor([[0, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.long),
        },
    ])
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = MetaModel(
        task_config=task_config,
        input_channels=["sig"],
        models=[MetaModelEntry(DummyEmbeddingModel(1, 1), ["sig"])],
    )
    with torch.no_grad():
        model.model_entries[0]["model"].proj.weight.zero_()
        model.heads["coarse"].weight.zero_()
        model.heads["coarse"].bias.copy_(torch.tensor([1.0, 0.0]))
        model.heads["sleep"].weight.zero_()
        model.heads["sleep"].bias.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))
        model.heads["breathing"].weight.zero_()
        model.heads["breathing"].bias.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))

    masked_trainer = MultiLabelTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        task_config={task: {**cfg, "loss_function": torch.nn.functional.cross_entropy} for task, cfg in task_config.items()},
        condition_task="sleep",
        condition_labels=["n2"],
        conditioned_tasks=["breathing"],
        device="cpu",
        warmup_device="cpu",
    )

    _, masked_cm = masked_trainer.run_epoch(loader, None, model, "TEST")

    assert masked_cm["breathing"].tolist() == [[2, 0], [1, 0]]


def test_estimate_class_cnts_respects_conditioning():
    task_config = MultiLabelTrainer.normalize_task_config({
        "sleep": {"labels": ["wake", "n2"], "default": None, "target_resolution": "30s"},
        "breathing": {
            "labels": ["apnea", "regular"],
            "default": "regular",
            "target_resolution": "10s",
            "loss_mode": "inverse-log",
        },
    })
    dataset = build_dataset([
        {
            "data": torch.zeros(4, 1),
            "target": torch.tensor([[1, -1, -1], [0, 1, 0]], dtype=torch.long),
        },
        {
            "data": torch.zeros(4, 1),
            "target": torch.tensor([[0, -1, -1], [1, 1, 1]], dtype=torch.long),
        },
    ])

    class_cnts = estimate_multilabel_class_cnts(
        dataset,
        task_config,
        condition_task="sleep",
        condition_labels=["n2"],
        conditioned_tasks=["breathing"],
        num_workers=0,
        batch_size=2,
    )

    assert class_cnts == {
        "sleep": {"wake": 1.0, "n2": 1.0},
        "breathing": {"apnea": 2.0, "regular": 1.0},
    }


def test_configure_losses_uses_explicit_task_weights_when_given():
    task_config = MultiLabelTrainer.normalize_task_config({
        "sleep": {
            "labels": ["wake", "n2"],
            "default": None,
            "target_resolution": "30s",
            "class_weights": {"wake": 2.0, "n2": 5.0},
        },
        "breathing": {
            "labels": ["apnea", "regular"],
            "default": "regular",
            "target_resolution": "10s",
            "class_weights": {"apnea": 7.0, "regular": 3.0},
        },
    })
    configured = {}
    for task, cfg in task_config.items():
        current_cfg = dict(cfg)
        weights = torch.tensor([cfg["class_weights"][label] for label in cfg["labels"]], dtype=torch.float32)
        current_cfg["loss_function"] = partial(torch.nn.functional.cross_entropy, weight=weights)
        configured[task] = current_cfg

    sleep_weight = configured["sleep"]["loss_function"].keywords["weight"]
    breathing_weight = configured["breathing"]["loss_function"].keywords["weight"]

    assert torch.equal(sleep_weight.cpu(), torch.tensor([2.0, 5.0]))
    assert torch.equal(breathing_weight.cpu(), torch.tensor([7.0, 3.0]))


def test_configure_losses_derives_task_weights_from_class_counts():
    task_config = MultiLabelTrainer.normalize_task_config({
        "sleep": {
            "labels": ["wake", "n2"],
            "default": None,
            "target_resolution": "30s",
            "loss_mode": "inverse",
        },
        "breathing": {
            "labels": ["apnea", "regular"],
            "default": "regular",
            "target_resolution": "10s",
            "loss_mode": "inverse",
        },
    })
    class_cnts = {
        "sleep": {"wake": 4.0, "n2": 1.0},
        "breathing": {"apnea": 1.0, "regular": 3.0},
    }
    configured = {}
    for task, cfg in task_config.items():
        current_cfg = dict(cfg)
        weights = class_weights_for_loss({}, class_cnts[task], cfg["loss_mode"])
        current_cfg["loss_function"] = partial(
            torch.nn.functional.cross_entropy,
            weight=torch.tensor([weights[label] for label in cfg["labels"]], dtype=torch.float32),
        )
        configured[task] = current_cfg

    sleep_weight = configured["sleep"]["loss_function"].keywords["weight"].cpu()
    breathing_weight = configured["breathing"]["loss_function"].keywords["weight"].cpu()

    assert sleep_weight[1] > sleep_weight[0]
    assert breathing_weight[0] > breathing_weight[1]


def test_prepare_target_respects_task_specific_class_counts(monkeypatch):
    task_config = MultiLabelTrainer.normalize_task_config({
        "sleep": {"labels": ["wake", "n2"], "default": None, "target_resolution": "30s"},
        "breathing": {"labels": ["apnea", "regular"], "default": "regular", "target_resolution": "10s"},
    })
    monkeypatch.setattr(
        MultiLabelTrainer,
        "build_multitask_target",
        staticmethod(lambda targets, task_config, raise_error=False: torch.tensor([[1, -1, -1], [0, 1, 0]], dtype=torch.long)),
    )

    random_values = iter([0.1, 0.3])
    monkeypatch.setattr("sleepwalker.trainer.MultiLabelTrainer.random.random", lambda: next(random_values))

    keep = MultiLabelTrainer.prepare_target(
        target=pd.DataFrame(),
        task_config=task_config,
        class_cnts={
            "sleep": {"wake": 8.0, "n2": 2.0},
            "breathing": {"apnea": 1.0, "regular": 9.0},
        },
    )
    drop = MultiLabelTrainer.prepare_target(
        target=pd.DataFrame(),
        task_config=task_config,
        class_cnts={
            "sleep": {"wake": 8.0, "n2": 2.0},
            "breathing": {"apnea": 1.0, "regular": 9.0},
        },
    )

    assert keep is not None
    assert drop is None
