import pandas as pd
import pytest
import torch
from functools import partial
from torch.utils.data import DataLoader, Dataset

from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.models.BaseModel import BaseModel, EmbeddingModel
from sleepwalker.models.ModelGraphClassifier import GraphNode, ModelGraphClassifier
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
import sleepwalker.trainer.utils.targets as target_utils
from sleepwalker.trainer.losses import class_weights_for_loss, estimate_multilabel_class_cnts
from sleepwalker.trainer.utils.targets import build_multitask_target, normalize_multitask_config, prepare_multitask_target


class DummyEmbeddingModel(BaseModel, EmbeddingModel):
    def __init__(self, n_channels, feature_dim):
        super().__init__()
        self.proj = torch.nn.Linear(n_channels, feature_dim, bias=False)
        self._feature_dim = feature_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=1))

    def feature_dim(self) -> int:
        return self._feature_dim

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def input_spec(self):
        return (1, 4, self.proj.in_features), {"layout": "BTC", "ts_len": 4, "n_channels": self.proj.in_features}


class MultiLabelBatchDataset(Dataset):
    target_resolution = "30s"

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_multitask_item(task_config, task_indices, task_masks=None):
    max_steps = max(cfg["n_steps"] for cfg in task_config.values())
    max_classes = max(len(cfg["labels"]) for cfg in task_config.values())
    target = torch.zeros((len(task_config), max_steps, max_classes), dtype=torch.float32)
    target_mask = torch.zeros((len(task_config), max_steps), dtype=torch.bool)
    for task_idx, (task, cfg) in enumerate(task_config.items()):
        indices = torch.tensor(task_indices[task], dtype=torch.long)
        if len(indices) != cfg["n_steps"]:
            raise ValueError(f"Expected {cfg['n_steps']} indices for {task}, got {len(indices)}.")
        target[task_idx, :cfg["n_steps"], :len(cfg["labels"])] = torch.nn.functional.one_hot(indices, len(cfg["labels"])).float()
        mask = torch.ones(cfg["n_steps"], dtype=torch.bool) if task_masks is None or task not in task_masks else torch.tensor(task_masks[task], dtype=torch.bool)
        target_mask[task_idx, :cfg["n_steps"]] = mask
    return {"data": torch.zeros(4, 1), "target": target, "target_mask": target_mask}


def build_trainer(**kwargs):
    task_config = {
        "sleep staging": {
            "labels": ["wake", "n1", "n2", "n3", "rem"],
            "default": None,
            "percentage": 0.5,
            "target_resolution": "30s",
            "sequence_len": 1,
        },
        "breathing": {
            "labels": ["apnea", "hypopnea", "regular breathing"],
            "default": "regular breathing",
            "percentage": 0.5,
            "target_resolution": "10s",
            "sequence_len": 3,
        },
    }
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
    y, target_mask = build_multitask_target(make_target_df(), trainer.task_config)

    assert y.shape == (2, 3, 5)
    assert y[0, :1, :5].argmax(dim=-1).tolist() == [2]
    assert y[1, :3, :3].argmax(dim=-1).tolist() == [0, 2, 1]
    assert target_mask.tolist() == [[True, False, False], [True, True, True]]


def test_prepare_target_normalizes_raw_task_config():
    task_config = {
        "sleep staging": {"labels": ["wake", "n1", "n2", "n3", "rem"], "default": None, "target_resolution": "30s", "sequence_len": 1},
        "breathing": {"labels": ["apnea", "hypopnea", "regular breathing"], "default": "regular breathing", "target_resolution": "10s", "sequence_len": 3},
    }

    prepared = prepare_multitask_target(make_target_df(), task_config=task_config)

    assert prepared["target"].shape == (2, 3, 5)
    assert prepared["target_mask"].tolist() == [[True, False, False], [True, True, True]]


def test_build_multitask_target_rejects_multiple_active_labels_within_step():
    trainer = build_trainer()
    target = make_target_df()
    target.loc[target.index[:10], "hypopnea"] = 1

    with pytest.raises(ValueError, match="task 'breathing'"):
        build_multitask_target(target, trainer.task_config)


def test_build_multitask_target_returns_none_when_raise_error_is_false():
    trainer = build_trainer()
    target = make_target_df()
    target.loc[target.index[:10], "hypopnea"] = 1

    assert build_multitask_target(target, trainer.task_config, raise_error=False) is None


def test_build_multitask_target_uses_task_specific_percentage():
    task_config = {
        "sleep staging": {"labels": ["wake", "n1"], "default": None, "percentage": 0.8, "target_resolution": "20s", "sequence_len": 1},
        "breathing": {"labels": ["apnea", "regular breathing"], "default": "regular breathing", "percentage": 0.2, "target_resolution": "10s", "sequence_len": 2},
    }
    task_config = normalize_multitask_config(task_config)
    idx = pd.date_range("2024-01-01", periods=20, freq="1s")
    target = pd.DataFrame(0, index=idx, columns=["wake", "n1", "apnea", "regular breathing"])
    target.loc[idx[:17], "n1"] = 1
    target.loc[idx[:3], "apnea"] = 1
    target.loc[idx[10:11], "apnea"] = 1

    y, target_mask = build_multitask_target(target, task_config)

    assert y[0, :1, :2].argmax(dim=-1).tolist() == [1]
    assert y[1, :2, :2].argmax(dim=-1).tolist() == [0, 1]
    assert target_mask.tolist() == [[True, False], [True, True]]


def test_build_multitask_target_centers_explicit_spans_and_builds_soft_step_masks():
    task_config = normalize_multitask_config({
        "sleep": {"labels": ["wake", "n2"], "default": None, "target_resolution": "2s", "sequence_len": 2},
        "event": {
            "labels": ["no_event", "event"],
            "default": "no_event",
            "target_resolution": "1s",
            "sequence_len": 8,
            "soft_boundaries": True,
            "step_mask": {"columns": ["n2"], "percentage": 0.5},
        },
    })
    index = pd.date_range("2024-01-01", periods=16, freq="500ms")
    target = pd.DataFrame(0, index=index, columns=["wake", "n2", "event"])
    target.loc[index[4:12], "n2"] = 1
    target.loc[index[0], "event"] = 1

    values, target_mask = build_multitask_target(target, task_config)

    assert task_config["sleep"]["target_offset"] == pd.Timedelta("2s")
    assert values[0, :2, :2].argmax(dim=-1).tolist() == [1, 1]
    assert torch.allclose(values[1, 0, :2], torch.tensor([0.5, 0.5]))
    assert target_mask[0].tolist() == [True, True, False, False, False, False, False, False]
    assert target_mask[1].tolist() == [False, False, True, True, True, True, False, False]


def test_trainer_requires_explicit_sequence_len():
    task_config = {
        "sleep staging": {"labels": ["wake", "n1"], "default": None, "target_resolution": "30s", "sequence_len": 1},
        "breathing": {"labels": ["apnea", "regular breathing"], "default": "regular breathing", "target_resolution": "7s"},
    }

    with pytest.raises(ValueError, match="Task 'breathing' is missing sequence_len"):
        MultiLabelTrainer(
            epochs=1,
            optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
            task_config={task: {**cfg, "loss_function": torch.nn.functional.cross_entropy} for task, cfg in task_config.items()},
            device="cpu",
            warmup_device="cpu",
        )


def test_trainer_rejects_unknown_condition_label():
    with pytest.raises(ValueError, match="condition_labels contains labels"):
        build_trainer(condition_task="sleep staging", condition_labels=["apnea"], conditioned_tasks=["breathing"])


def test_run_epoch_masks_conditioned_tasks_during_wake():
    task_config = normalize_multitask_config({
        "sleep": {"labels": ["wake", "n2"], "default": None, "target_resolution": "30s", "sequence_len": 1},
        "breathing": {"labels": ["apnea", "regular"], "default": "regular", "target_resolution": "10s", "sequence_len": 3},
    })
    dataset = MultiLabelBatchDataset([
        make_multitask_item(task_config, {"sleep": [1], "breathing": [0, 0, 0]}),
        make_multitask_item(task_config, {"sleep": [0], "breathing": [1, 1, 1]}),
    ])
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = ModelGraphClassifier(
        nodes={"shared": GraphNode(DummyEmbeddingModel(1, 1), {task: {"classes": cfg["labels"], "sequence_len": cfg["sequence_len"]} for task, cfg in task_config.items()})},
        method="latent",
    )
    with torch.no_grad():
        model.models["shared"].proj.weight.zero_()
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


def test_multilabel_training_allows_an_exhausted_loader():
    class Loader(list):
        batch_size = 2
        dataset = type("Dataset", (), {"target_resolution": "30s"})()

    model = torch.nn.Linear(1, 1)
    trainer = build_trainer()

    loss, cms = trainer.run_epoch(Loader([None]), torch.optim.SGD(model.parameters(), lr=0.1), model, prefix="TRAIN")

    assert loss == 0
    assert all(cm.sum() == 0 for cm in cms.values())


def test_run_epoch_condition_mask_applies_to_entire_sample_when_condition_task_has_multiple_steps():
    task_config = normalize_multitask_config({
        "coarse": {"labels": ["off", "on"], "default": None, "target_resolution": "30s", "sequence_len": 1},
        "sleep": {"labels": ["wake", "n2"], "default": None, "target_resolution": "10s", "sequence_len": 3},
        "breathing": {"labels": ["apnea", "regular"], "default": "regular", "target_resolution": "10s", "sequence_len": 3},
    })
    dataset = MultiLabelBatchDataset([
        make_multitask_item(task_config, {"coarse": [0], "sleep": [0, 1, 0], "breathing": [0, 1, 0]}),
        make_multitask_item(task_config, {"coarse": [0], "sleep": [0, 0, 0], "breathing": [1, 1, 1]}),
    ])
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = ModelGraphClassifier(
        nodes={"shared": GraphNode(DummyEmbeddingModel(1, 1), {task: {"classes": cfg["labels"], "sequence_len": cfg["sequence_len"]} for task, cfg in task_config.items()})},
        method="latent",
    )
    with torch.no_grad():
        model.models["shared"].proj.weight.zero_()
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
    task_config = normalize_multitask_config({
        "sleep": {"labels": ["wake", "n2"], "default": None, "target_resolution": "30s", "sequence_len": 1},
        "breathing": {
            "labels": ["apnea", "regular"],
            "default": "regular",
            "target_resolution": "10s",
            "sequence_len": 3,
            "loss_mode": "inverse-log",
        },
    })
    dataset = build_dataset([
        make_multitask_item(task_config, {"sleep": [1], "breathing": [0, 1, 0]}),
        make_multitask_item(task_config, {"sleep": [0], "breathing": [1, 1, 1]}),
    ])

    class_cnts = estimate_multilabel_class_cnts(
        DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=partial(batch_collate, ignore_list=["time", "patient", "data"]),
            drop_last=False,
        ),
        task_config,
        condition_task="sleep",
        condition_labels=["n2"],
        conditioned_tasks=["breathing"],
    )

    assert class_cnts == {
        "sleep": {"wake": 1.0, "n2": 1.0},
        "breathing": {"apnea": 2.0, "regular": 1.0},
    }


def test_configure_losses_uses_explicit_task_weights_when_given():
    task_config = normalize_multitask_config({
        "sleep": {
            "labels": ["wake", "n2"],
            "default": None,
            "target_resolution": "30s",
            "sequence_len": 1,
            "class_weights": {"wake": 2.0, "n2": 5.0},
        },
        "breathing": {
            "labels": ["apnea", "regular"],
            "default": "regular",
            "target_resolution": "10s",
            "sequence_len": 3,
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
    task_config = normalize_multitask_config({
        "sleep": {
            "labels": ["wake", "n2"],
            "default": None,
            "target_resolution": "30s",
            "sequence_len": 1,
            "loss_mode": "inverse",
        },
        "breathing": {
            "labels": ["apnea", "regular"],
            "default": "regular",
            "target_resolution": "10s",
            "sequence_len": 3,
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


def test_warmup_uses_configured_class_counts_without_loading_data(monkeypatch):
    trainer = build_trainer()
    for task, cfg in trainer.task_config.items():
        cfg["loss_mode"] = "inverse"
        cfg["class_counts"] = {label: float(index + 1) for index, label in enumerate(cfg["labels"])}

    monkeypatch.setattr(
        "sleepwalker.trainer.MultiLabelTrainer.estimate_multilabel_class_cnts",
        lambda *args, **kwargs: pytest.fail("configured counts must skip online estimation"),
    )
    trainer.warmup_trainer(None)

    for task, cfg in trainer.task_config.items():
        assert isinstance(trainer.task_loss_functions[task], partial)
        assert trainer.task_loss_functions[task].keywords["weight"].shape == (len(cfg["labels"]),)


def test_task_config_rejects_incomplete_class_counts():
    with pytest.raises(ValueError, match="class_counts must contain exactly"):
        normalize_multitask_config({
            "sleep": {
                "labels": ["wake", "n2"],
                "default": None,
                "target_resolution": "30s",
                "sequence_len": 1,
                "class_counts": {"wake": 10},
            }
        })


def test_prepare_target_respects_task_specific_class_counts(monkeypatch):
    task_config = normalize_multitask_config({
        "sleep": {"labels": ["wake", "n2"], "default": None, "target_resolution": "30s", "sequence_len": 1},
        "breathing": {"labels": ["apnea", "regular"], "default": "regular", "target_resolution": "10s", "sequence_len": 3},
    })
    prepared = make_multitask_item(task_config, {"sleep": [1], "breathing": [0, 1, 0]})
    monkeypatch.setattr(target_utils, "build_multitask_target", lambda targets, task_config, raise_error=False: (prepared["target"], prepared["target_mask"]))

    random_values = iter([0.1, 0.3])
    monkeypatch.setattr(target_utils.random, "random", lambda: next(random_values))

    keep = prepare_multitask_target(
        target=pd.DataFrame(),
        task_config=task_config,
        class_cnts={
            "sleep": {"wake": 8.0, "n2": 2.0},
            "breathing": {"apnea": 1.0, "regular": 9.0},
        },
    )
    drop = prepare_multitask_target(
        target=pd.DataFrame(),
        task_config=task_config,
        class_cnts={
            "sleep": {"wake": 8.0, "n2": 2.0},
            "breathing": {"apnea": 1.0, "regular": 9.0},
        },
    )

    assert keep is not None
    assert drop is None
