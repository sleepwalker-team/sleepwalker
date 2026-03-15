import pytest
import torch

from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer


def build_trainer():
    task_config = {
        "sleep staging": {"labels": ["wake", "n1", "n2", "n3", "rem"], "default": None, "percentage": 0.5},
        "breathing": {"labels": ["apnea", "hypopnea", "regular breathing"], "default": "regular breathing", "percentage": 0.5},
    }
    return MultiLabelTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        task_config=task_config,
        loss_function=torch.nn.functional.cross_entropy,
        device="cpu",
        warmup_device="cpu",
    )


def test_trainer_flattens_task_config():
    trainer = build_trainer()

    assert trainer.classes == ["wake", "n1", "n2", "n3", "rem", "apnea", "hypopnea", "regular breathing"]
    assert trainer.task_slices["sleep staging"] == slice(0, 5)
    assert trainer.task_slices["breathing"] == slice(5, 8)


def test_target_to_multilabel_uses_placeholder_per_task():
    trainer = build_trainer()

    y = trainer.target_to_multilabel(
        {
            "n2": 20.0,
            "apnea": 0.0,
            "hypopnea": 0.0,
        },
        trainer.task_config,
        20.0,
    )

    assert y.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_target_to_multilabel_rejects_multiple_active_labels_within_task():
    trainer = build_trainer()

    with pytest.raises(ValueError, match="Multiple active classes found for task 'breathing'"):
        trainer.target_to_multilabel(
            {"n2": 20.0, "apnea": 20.0, "hypopnea": 20.0},
            trainer.task_config,
            20.0,
        )


def test_target_to_multilabel_uses_task_specific_percentage():
    task_config = {
        "sleep staging": {"labels": ["wake", "n1"], "default": None, "percentage": 0.8},
        "breathing": {"labels": ["apnea", "regular breathing"], "default": "regular breathing", "percentage": 0.2},
    }

    y = MultiLabelTrainer.target_to_multilabel(
        {"n1": 17.0, "apnea": 3.0},
        task_config,
        20.0,
    )

    assert y.tolist() == [0.0, 1.0, 0.0, 1.0]
