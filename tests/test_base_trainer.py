from functools import partial
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from sleepwalker.trainer.BaseTrainer import BaseTrainer
from sleepwalker.trainer.Run import export_final_model


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def warmup_preprocessor(self, data_loader, device):
        return self


class ScalarTrainer(BaseTrainer):
    def classification_contract(self):
        return {"type": "test"}

    def run_epoch(self, loader, opt, model, prefix="", lr_scheduler=None):
        if opt is not None:
            with torch.no_grad():
                model.value.add_(1)
            return float(model.value.item()), {}
        return abs(float(model.value.item()) - 1.0), {}


def make_trainer(*, return_best: bool, epochs: int = 3, save_every: int = 0):
    return ScalarTrainer(
        epochs=epochs,
        optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        device="cpu",
        warmup_device="cpu",
        save_every=save_every,
        eval_every=1,
        return_best=return_best,
    )


def make_loader():
    return DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)


def test_return_best_exposes_lowest_validation_checkpoint():
    model = ScalarModel()
    result = make_trainer(return_best=True).fit(model, make_loader(), make_loader())

    assert result["best_model"] == 0
    assert result["best_model_state"]["value"].item() == 1.0
    assert model.value.item() == 3.0


def test_return_last_leaves_model_at_final_epoch_without_reload_checkpoint():
    model = ScalarModel()
    result = make_trainer(return_best=False).fit(model, make_loader(), make_loader())

    assert "best_model_state" not in result
    assert "best_model" not in result
    assert model.value.item() == 3.0


def test_checkpoint_callback_uses_one_indexed_save_every_cadence(monkeypatch):
    model = ScalarModel()
    trainer = make_trainer(return_best=False, epochs=5, save_every=2)
    training_checkpoints = []

    def log_artifact(path, dest=None):
        checkpoint_trainer, checkpoint_model = BaseTrainer.load_checkpoint(path)
        training_checkpoints.append((checkpoint_trainer.completed_epochs, checkpoint_model.value.item(), dest))

    monkeypatch.setattr("sleepwalker.trainer.BaseTrainer.logger.artifact", log_artifact)

    trainer.fit(model, make_loader())

    assert training_checkpoints == [(2, 2.0, "2"), (4, 4.0, "4")]


def test_default_checkpoint_and_evaluation_cadence_are_ten():
    trainer = ScalarTrainer(epochs=1, optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")

    assert trainer.save_every == 10
    assert trainer.eval_every == 10


def test_run_exports_the_training_dataset(monkeypatch):
    captured = {}
    trainer = make_trainer(return_best=False)

    def save_package(path, **kwargs):
        package_path = Path(path)
        package_path.mkdir(parents=True)
        (package_path / "manifest.json").write_text("{}\n", encoding="utf-8")
        captured["save"] = kwargs

    def log_artifact(path, dest=None):
        assert (Path(path) / "manifest.json").is_file()
        captured["dest"] = dest

    monkeypatch.setattr("sleepwalker.trainer.Run.save_packaged_model", save_package)
    monkeypatch.setattr("sleepwalker.trainer.Run.logger.artifact", log_artifact)
    dataset = object()
    cfg = SimpleNamespace(experiment_name="model", expert_task="task", model_name="model", model=ScalarModel(), trainer=trainer, meta_data={"seed": 17}, package_path=None)

    export_final_model(cfg, dataset)

    assert captured["save"]["dataset"] is dataset
    assert captured["save"]["name"] == "model"
    assert captured["save"]["classification_contract"] == {"type": "test"}
    assert captured["dest"] == "final"


def test_run_can_export_a_stable_package(monkeypatch, tmp_path):
    captured = {}
    trainer = make_trainer(return_best=False)

    class Package:
        def save(self, path):
            captured["package_path"] = path

    def save_package(path, **kwargs):
        package_path = Path(path)
        package_path.mkdir(parents=True)
        (package_path / "manifest.json").write_text("{}\n", encoding="utf-8")
        return Package()

    monkeypatch.setattr("sleepwalker.trainer.Run.save_packaged_model", save_package)
    monkeypatch.setattr("sleepwalker.trainer.Run.logger.artifact", lambda path, dest=None: None)
    monkeypatch.setattr("sleepwalker.trainer.Run.logger.info", lambda message: captured.setdefault("message", message))
    target = tmp_path / "stable-package"
    cfg = SimpleNamespace(experiment_name="model", expert_task="task", model_name="model", model=ScalarModel(), trainer=trainer, meta_data={"seed": 17}, package_path=str(target))

    export_final_model(cfg, object())

    assert captured["package_path"] == str(target)
    assert str(target) in captured["message"]


class OptimizingScalarTrainer(BaseTrainer):
    def classification_contract(self):
        return {"type": "test"}

    def run_epoch(self, loader, opt, model, prefix="", lr_scheduler=None):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss = (model.value - 3.0).square()
            loss.backward()
            opt.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
        else:
            loss = (model.value - 3.0).square()
        self.steps["train" if opt is not None else "val"] += 1
        return float(loss.item()), {}


def make_optimizing_trainer():
    return OptimizingScalarTrainer(
        epochs=6,
        optimizer=lambda model: torch.optim.AdamW(model.parameters(), lr=0.01),
        lr_scheduler=partial(torch.optim.lr_scheduler.OneCycleLR, max_lr=0.1, pct_start=0.5),
        device="cpu",
        warmup_device="cpu",
        save_every=3,
        return_best=False,
    )


def assert_nested_equal(first, second):
    assert type(first) is type(second)
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for first_item, second_item in zip(first, second):
            assert_nested_equal(first_item, second_item)
    elif isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    else:
        assert first == second


def test_resume_matches_uninterrupted_optimizer_and_scheduler_state(tmp_path, monkeypatch):
    destination = tmp_path / "artifacts"
    run_label = ["full"]

    def log_artifact(path, dest=None):
        source = Path(path)
        if source.name == "checkpoint.pt":
            destination.mkdir(exist_ok=True)
            shutil.copy2(source, destination / f"{run_label[0]}-{dest}.pt")

    monkeypatch.setattr("sleepwalker.trainer.BaseTrainer.logger.artifact", log_artifact)
    full_model = ScalarModel()
    full_trainer = make_optimizing_trainer()
    full_trainer.fit(full_model, make_loader())

    run_label[0] = "resumed"
    resumed_trainer, resumed_model = BaseTrainer.load_checkpoint(destination / "full-3.pt")
    assert resumed_trainer.optimizer.param_groups[0]["params"][0] is resumed_model.value
    assert resumed_trainer.lr_scheduler.optimizer is resumed_trainer.optimizer
    with pytest.raises(RuntimeError, match="Optimizer parameters do not belong"):
        resumed_trainer.fit(ScalarModel(), make_loader())
    resumed_trainer.fit(resumed_model, make_loader())

    full_checkpoint, full_checkpoint_model = BaseTrainer.load_checkpoint(destination / "full-6.pt")
    resumed_checkpoint, resumed_checkpoint_model = BaseTrainer.load_checkpoint(destination / "resumed-6.pt")
    assert_nested_equal(full_checkpoint_model.state_dict(), resumed_checkpoint_model.state_dict())
    assert_nested_equal(full_checkpoint.optimizer.state_dict(), resumed_checkpoint.optimizer.state_dict())
    assert_nested_equal(full_checkpoint.lr_scheduler.state_dict(), resumed_checkpoint.lr_scheduler.state_dict())
    assert full_checkpoint.steps == resumed_checkpoint.steps
    assert full_checkpoint.epoch_step == resumed_checkpoint.epoch_step
