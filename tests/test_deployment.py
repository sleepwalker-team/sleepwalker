import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from sleepwalker.datasets.Basedataset import BaseDataset, ChannelConfig, batch_collate
from sleepwalker.datasets.NumpyDataset import NumpyDataset
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.deployment import export_prediction_package, load_prediction_package


class TinyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2, bias=True)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def feature_dim(self) -> int:
        return 1

    def _classifier(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TinyWindowDataset(Dataset):
    def __init__(self):
        self.items = [
            {
                "data": torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
                "patient": "p1.edf",
                "time": pd.Timestamp("2024-01-01 00:00:00"),
            },
            {
                "data": torch.tensor([[1.0], [1.0], [1.0], [1.0]], dtype=torch.float32),
                "patient": "p1.edf",
                "time": pd.Timestamp("2024-01-01 00:00:30"),
            },
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class MinimalDataset(BaseDataset):
    def __init__(self):
        super().__init__(
            channels=[ChannelConfig(name="sig", normalizer=None)],
            sample_frequency=1.0,
            total_input="4s",
            target_resolution="2s",
            event_mapping=None,
        )

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError


class RunDataset(Dataset):
    def __init__(self):
        self.channels = [ChannelConfig(name="sig", normalizer=None)]
        self.items = [{"data": torch.zeros(4, 1), "target": torch.tensor([1.0, 0.0])}]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def get_n_patients(self):
        return 1

    def get_classes(self):
        return ["neg", "pos"]

    def get_timeseries_len(self):
        return 4


class StubTrainer:
    def fit(self, model, train_loader, val_loader=None):
        return {}

    def test(self, model, test_loader):
        return 0.0, {}


def build_trainer():
    return MulticlassTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        classes=["neg", "pos"],
        loss_function=torch.nn.functional.cross_entropy,
        device="cpu",
        warmup_device="cpu",
    )


def test_multiclass_predict_loader_returns_timestamped_dataframe():
    trainer = build_trainer()
    model = TinyModel()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[-1.0], [1.0]]))
        model.linear.bias.zero_()

    loader = DataLoader(TinyWindowDataset(), batch_size=2, shuffle=False, collate_fn=batch_collate)
    result = trainer.predict_loader(model, loader)

    assert list(result["patient"]) == ["p1.edf", "p1.edf"]
    assert list(result["prediction"]) == ["neg", "pos"]
    assert "prob__neg" in result.columns
    assert "prob__pos" in result.columns


def test_prediction_package_roundtrip_preserves_trainer_and_dataset_template():
    trainer = build_trainer()
    model = TinyModel()
    dataset = MinimalDataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = Path(tmpdir) / "test.swmodel"
        export_prediction_package(
            str(package_path),
            model=model,
            trainer=trainer,
            dataset=dataset,
            model_card_md="# test\n",
            metadata={"experiment": "unit-test"},
        )
        assert package_path.is_file()
        package = load_prediction_package(str(package_path), map_location="cpu")

        assert package.metadata["experiment"] == "unit-test"
        assert package.metadata["source_event_mapping"] == {}
        assert isinstance(package.unlabelled_dataset, UnlabelledDataset)
        assert package.unlabelled_dataset.initialized is False
        assert package.trainer.__class__.__name__ == "MulticlassTrainer"
        assert package.model.__class__.__name__ == "TinyModel"


def test_export_prediction_package_appends_swmodel_extension_when_missing():
    trainer = build_trainer()
    model = TinyModel()
    dataset = MinimalDataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = Path(tmpdir) / "test_package"
        export_prediction_package(
            str(package_path),
            model=model,
            trainer=trainer,
            dataset=dataset,
        )
        assert (Path(tmpdir) / "test_package.swmodel").is_file()


def test_run_writes_meta_data_into_run_folder():
    model = TinyModel()
    trainer = StubTrainer()
    train_dataset = RunDataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run(
            RunCfg(
                experiment_name="unit-run-meta",
                model_name="TinyModel",
                model=model,
                trainer=trainer,
                train_datasets=[train_dataset],
                val_datasets=[],
                test_datasets=[],
                batch_size=1,
                n_samples=None,
                num_workers_dataloader=0,
                collate_fn=batch_collate,
                log_path=tmpdir,
                meta_data={"source": "run", "kind": "test"},
            )
        )

        meta_path = Path(tmpdir) / "unit-run-meta" / "meta_data.yml"
        assert result.experiment_name == "unit-run-meta"
        assert meta_path.is_file()
        content = meta_path.read_text(encoding="utf-8")
        assert "source" in content
        assert "run" in content


def test_export_prediction_package_accepts_numpy_dataset_and_exports_unlabelled_dataset():
    model = TinyModel()
    trainer = build_trainer()

    with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as export_dir:
        cache_path = Path(cache_dir)
        with (cache_path / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "sample_frequency": 1.0,
                    "resample_type": "nearest",
                    "total_input": "4s",
                    "target_resolution": "2s",
                    "classes": ["neg", "pos"],
                    "input_channels": ["sig"],
                    "all_patients": ["p1.edf"],
                },
                f,
            )
        np.save(cache_path / "data.npy", np.zeros((1, 4, 1), dtype=np.float32))
        np.save(cache_path / "patient.npy", np.array(["p1.edf"]))
        np.save(cache_path / "time.npy", np.array([pd.Timestamp("2024-01-01").value], dtype=np.int64))

        dataset = NumpyDataset(cache_path, in_memory=True)
        package_path = os.path.join(export_dir, "numpy.swmodel")
        package = export_prediction_package(
            package_path,
            model=model,
            trainer=trainer,
            dataset=dataset,
        )
        assert os.path.isfile(package_path)
        assert isinstance(package.unlabelled_dataset, UnlabelledDataset)
