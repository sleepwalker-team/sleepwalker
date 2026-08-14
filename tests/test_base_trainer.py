from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader, TensorDataset

from sleepwalker.trainer.BaseTrainer import BaseTrainer


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
        return_best=return_best,
    )


def make_loader():
    return DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)


def test_return_best_exposes_lowest_validation_checkpoint():
    model = ScalarModel()
    result = make_trainer(return_best=True).fit(model, make_loader(), make_loader())

    state = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    assert result["best_model"] == 0
    assert state["value"].item() == 1.0
    assert model.value.item() == 3.0
    shutil.rmtree(Path(result["checkpoint"]).parent)


def test_return_last_leaves_model_at_final_epoch_without_reload_checkpoint():
    model = ScalarModel()
    result = make_trainer(return_best=False).fit(model, make_loader(), make_loader())

    assert "checkpoint" not in result
    assert "best_model" not in result
    assert model.value.item() == 3.0


def test_checkpoint_callback_uses_one_indexed_save_every_cadence():
    model = ScalarModel()
    trainer = make_trainer(return_best=False, epochs=5, save_every=2)
    checkpoints = []
    trainer.export_model = lambda current_model, dataset, **kwargs: checkpoints.append((kwargs["config"]["checkpoint_epoch"], current_model.value.item()))

    trainer.fit(model, make_loader())

    assert checkpoints == [(2, 2.0), (4, 4.0)]


def test_export_model_packages_the_training_dataset(monkeypatch):
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

    monkeypatch.setattr("sleepwalker.trainer.BaseTrainer.save_packaged_model", save_package)
    monkeypatch.setattr("sleepwalker.trainer.BaseTrainer.logger.artifact", log_artifact)
    dataset = object()

    trainer.export_model(ScalarModel(), dataset, name="model", task="task", config={"seed": 17}, dest="10")

    assert captured["save"]["dataset"] is dataset
    assert captured["save"]["name"] == "model"
    assert captured["save"]["classification_contract"] == {"type": "test"}
    assert captured["dest"] == "10"
