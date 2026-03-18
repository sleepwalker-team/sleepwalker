import pandas as pd
import pytest
import torch

from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer


def build_trainer():
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
    return MultiLabelTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        task_config=task_config,
        loss_function=torch.nn.functional.cross_entropy,
        device="cpu",
        warmup_device="cpu",
    )


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


def test_target_to_multiclass_uses_task_steps():
    trainer = build_trainer()
    y = trainer.target_to_multiclass(make_target_df(), trainer.task_config)

    assert y.shape == (2, 3)
    assert y[0].tolist() == [2, -1, -1]
    assert y[1].tolist() == [0, 2, 1]


def test_target_to_multiclass_rejects_multiple_active_labels_within_step():
    trainer = build_trainer()
    target = make_target_df()
    target.loc[target.index[:10], "hypopnea"] = 1

    with pytest.raises(ValueError, match="Task 'breathing', step 0"):
        trainer.target_to_multiclass(target, trainer.task_config)


def test_target_to_multiclass_returns_none_when_raise_error_is_false():
    trainer = build_trainer()
    target = make_target_df()
    target.loc[target.index[:10], "hypopnea"] = 1

    assert trainer.target_to_multiclass(target, trainer.task_config, raise_error=False) is None


def test_target_to_multiclass_uses_task_specific_percentage():
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

    y = MultiLabelTrainer.target_to_multiclass(target, task_config)

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
            task_config=MultiLabelTrainer.normalize_task_config(task_config),
            loss_function=torch.nn.functional.cross_entropy,
            device="cpu",
            warmup_device="cpu",
        )
