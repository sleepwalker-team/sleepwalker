from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyDataset:
    def get_n_patients(self):
        return 20


def load_estimator():
    path = REPO_ROOT / "iclr2026" / "scripts" / "estimate_class_distribution.py"
    spec = importlib.util.spec_from_file_location("iclr_class_distribution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_configs_use_one_representative_per_single_task():
    estimator = load_estimator()

    assert [path.name for path in estimator.DEFAULT_CONFIGS] == [
        "sleep_sleepwalker.yml",
        "arousal_sleepwalker.yml",
        "breathing_sleepwalker.yml",
        "desaturation_sleepwalker.yml",
    ]


def test_estimator_reuses_training_dataset_loader_and_count_helpers(monkeypatch, tmp_path):
    estimator = load_estimator()
    dataset = DummyDataset()
    calls = {}
    config = {
        "seed": 17,
        "data": {"num_workers": 1},
        "run": {
            "batch_size": 8,
            "num_workers_dataloader": 4,
            "patients_per_epoch": 32,
            "expert_task": "sleep",
        },
    }
    monkeypatch.setattr(estimator, "read_yaml", lambda path: config)
    monkeypatch.setattr(estimator, "initialize_datasets", lambda config: ([dataset], [], []))
    monkeypatch.setattr(estimator, "combine_datasets", lambda datasets: datasets[0])
    monkeypatch.setattr(estimator, "build_loader", lambda current_dataset, **kwargs: calls.update(dataset=current_dataset, loader=kwargs) or "loader")
    monkeypatch.setattr(estimator, "estimate_class_cnts", lambda loader: {"wake": 10.0})
    result = estimator.estimate_config(tmp_path / "sleep.yml", samples=100, workers=3)

    assert config["data"]["num_workers"] == 1
    assert calls["dataset"] is dataset
    assert calls["loader"]["n_samples"] == 100
    assert calls["loader"]["num_workers"] == 4
    assert calls["loader"]["patients_per_epoch"] == 32
    assert result["class_counts"] == {"wake": 10.0}
