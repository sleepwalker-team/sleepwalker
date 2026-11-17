import torch
from torch.utils.data import DataLoader

from sleepwalker.trainer.MaskedAutoencoderTrainer import MaskedAutoencoderTrainer


class ScalarReconstructor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.preprocessors = torch.nn.ModuleDict({"EEG": torch.nn.ModuleList()})

    def apply_preprocessors(self, data, modality):
        return data

    def forward(self, data, modality):
        return data * self.weight, torch.ones_like(data)


def test_masked_autoencoder_training_saves_checkpoints_and_steps_scheduler(tmp_path, monkeypatch):
    directories = iter([tmp_path / "best-0", tmp_path / "epoch-1", tmp_path / "best-1"])
    monkeypatch.setattr("sleepwalker.trainer.MaskedAutoencoderTrainer.tempfile.mkdtemp", lambda prefix: str(next(directories)))
    samples = [{"data_EEG": torch.ones(2, 3, 1), "mask_EEG": torch.ones(1)}] * 2
    train_loader = DataLoader(samples, batch_size=1)
    val_loader = DataLoader(samples[:1], batch_size=1)
    model = ScalarReconstructor()
    trainer = MaskedAutoencoderTrainer(
        epochs=2,
        optimizer=lambda parameters: torch.optim.SGD(parameters.parameters(), lr=0.1),
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=4),
        classes=[],
        groups=["EEG"],
        device="cpu",
        warmup_device="cpu",
        save_every=1,
    )

    trainer.fit(model, train_loader, val_loader)

    assert trainer.steps == {"train": 4, "val": 2, "test": 0}
    assert model.weight.item() > 0
    assert (tmp_path / "best-1" / "model.pt").is_file()
    assert (tmp_path / "epoch-1" / "scheduler.pt").is_file()
