from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.trainer.BaseTrainer import BaseTrainer
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.trainer.utils.targets import prepare_multiclass_target, prepare_multitask_target
from sleepwalker.training.loader import build_loader


class WindowDataset(Dataset):
    def __init__(self, task_config=None):
        self.prepare_target_callback = partial(prepare_multiclass_target, target_classes=["regular", "event"], target_resolution="1s") if task_config is None else partial(prepare_multitask_target, task_config=task_config)

    def __len__(self):
        return 100

    def __getitem__(self, index):
        target = pd.DataFrame({"event": [float(index >= 90)]}, index=pd.date_range("2025-01-01", periods=1, freq="1s"))
        prepared = self.prepare_target_callback(target=target)
        return None if prepared is None else {"data": torch.zeros(1, 1), **prepared}


class ProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(2))
        self.warmup_calls = 0

    def warmup_preprocessor(self, loader, device):
        assert loader.num_workers == 1 and not loader.persistent_workers
        # This pass must see callback changes made after the count pass ended.
        self.warmup_rows = sum(len(batch["target"]) for batch in loader if batch is not None)
        self.warmup_calls += 1


class ObserveEpoch:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observations = []

    def run_epoch(self, loader, opt, model, prefix="", lr_scheduler=None):
        assert loader.num_workers == 1 and loader.persistent_workers
        counts = torch.zeros(2)
        for batch in loader:
            if batch is not None:
                counts += batch["target"].reshape(-1, 2).sum(0)
        self.observations.append(counts.tolist())
        return 0.0, np.diag(counts.numpy())


class ObserveMulticlass(ObserveEpoch, MulticlassTrainer):
    pass


class ObserveMultitask(ObserveEpoch, MultiLabelTrainer):
    pass


def loader(dataset):
    return build_loader(dataset, batch_size=10, num_workers=1, n_samples=None, collate_fn=batch_collate, sampling="sequential", seed=17)


@pytest.mark.parametrize("multitask", [False, True])
def test_parallel_warmup_configures_workers_and_resume_reuses_counts(tmp_path, monkeypatch, multitask):
    options = dict(epochs=1, optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.1), device="cpu", warmup_device="cpu", balance_batches=True, save_every=0, eval_every=0)
    tasks = {"breathing": {"labels": ["regular", "event"], "default": "regular", "sequence_len": 1, "target_resolution": "1s", "loss_function": torch.nn.functional.cross_entropy}}
    trainer = ObserveMultitask(task_config=tasks, **options) if multitask else ObserveMulticlass(classes=["regular", "event"], target_resolution="1s", loss_function=torch.nn.functional.cross_entropy, **options)
    model = ProbeModel()
    train_loader = loader(WindowDataset(tasks if multitask else None))
    trainer.fit(model, train_loader)
    counts = trainer.task_config["breathing"]["class_counts"] if multitask else trainer.class_counts
    assert counts == {"regular": 90, "event": 10}
    assert 10 <= model.warmup_rows < 50
    assert trainer.observations[-1][1] == 10
    assert trainer.observations[-1][0] < 40

    path = trainer.save_checkpoint(tmp_path / "checkpoint.pt", model)
    restored, model = BaseTrainer.load_checkpoint(path)
    restored.epochs = 2

    def unexpected_count_pass(*args, **kwargs):
        pytest.fail("Resume must reuse the saved training counts.")

    module = "sleepwalker.trainer.MultiLabelTrainer.estimate_multilabel_class_cnts" if multitask else "sleepwalker.trainer.MulticlassTrainer.estimate_class_cnts"
    monkeypatch.setattr(module, unexpected_count_pass)
    restored.fit(model, loader(WindowDataset(tasks if multitask else None)))
    assert model.warmup_calls == 1
    assert restored.observations[-1][1] == 10
    assert restored.observations[-1][0] < 40
