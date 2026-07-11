import json
from pathlib import Path
import sys
import tempfile

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepwalker.deployment import load_expert_package, save_expert_package
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.models.preprocessors.RobustScaler import RobustScaler
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run
from train_arousal import build_expert_components as build_arousal_expert_components


BUILDER_CALLS: list[dict] = []


class TinyModel(BaseModel):
    def __init__(self, *, ts_len: int, n_channels: int, classes: list[str], preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.ts_len = ts_len
        self.n_channels = n_channels
        self.classes = list(classes)
        self.linear = torch.nn.Linear(n_channels, len(classes), bias=True)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def feature_dim(self) -> int:
        return self.n_channels

    def _classifier(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def input_spec(self):
        return (
            (1, self.ts_len, self.n_channels),
            {"layout": "BTC", "ts_len": self.ts_len, "n_channels": self.n_channels},
        )


class TinyDatasetTemplate:
    def __init__(
        self,
        *,
        channels: list[str],
        sample_frequency: float,
        total_input: str,
        target_resolution: str,
        stride: str,
    ):
        self._channels = list(channels)
        self.sample_frequency = float(sample_frequency)
        self.total_input = pd.to_timedelta(total_input)
        self.target_resolution = pd.to_timedelta(target_resolution)
        self.stride = pd.to_timedelta(stride)
        self.resample_type = "nearest"
        self.rereference = None
        self.prepare_patient_callback = None
        self.prepare_target_callback = None
        self.prepare_sample_callback = None

    def get_input_channels(self):
        return list(self._channels)


class TinyTrainDataset:
    def __init__(self):
        self.items = [
            {"data": torch.zeros(4, 2), "target": torch.tensor([1.0, 0.0])},
            {"data": torch.ones(4, 2), "target": torch.tensor([0.0, 1.0])},
        ]
        self.sample_frequency = 1.0
        self.total_input = pd.to_timedelta("4s")
        self.target_resolution = pd.to_timedelta("2s")
        self.stride = pd.to_timedelta("2s")
        self.resample_type = "nearest"
        self.rereference = None
        self.prepare_patient_callback = None
        self.prepare_target_callback = None
        self.prepare_sample_callback = None

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

    def get_input_channels(self):
        return ["sig_a", "sig_b"]


def build_tiny_components(config: dict):
    BUILDER_CALLS.append(dict(config))
    classes = list(config.get("classes", ["neg", "pos"]))
    preprocessors = None
    if config.get("robust_scaler"):
        preprocessors = [RobustScaler(channels=list(config.get("scaler_channels", range(int(config["n_channels"])))))]
    model = TinyModel(
        ts_len=int(config["ts_len"]),
        n_channels=int(config["n_channels"]),
        classes=classes,
        preprocessors=preprocessors,
    )
    trainer = MulticlassTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3))),
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.5,
            total_iters=2,
        ),
        classes=classes,
        loss_function=torch.nn.functional.cross_entropy,
        device="cpu",
        warmup_device="cpu",
    )
    dataset_template = TinyDatasetTemplate(
        channels=list(config["channels"]),
        sample_frequency=float(config["sample_frequency"]),
        total_input=str(config["total_input"]),
        target_resolution=str(config["target_resolution"]),
        stride=str(config["stride"]),
    )
    return {
        "model": model,
        "trainer": trainer,
        "dataset_template": dataset_template,
    }


def _build_components():
    cfg = {
        "ts_len": 4,
        "n_channels": 2,
        "channels": ["sig_a", "sig_b"],
        "classes": ["neg", "pos"],
        "sample_frequency": 1.0,
        "total_input": "4s",
        "target_resolution": "2s",
        "stride": "2s",
        "lr": 1e-3,
    }
    return cfg, build_tiny_components(cfg)


def test_save_load_expert_package_roundtrip_preserves_manifest_and_weights():
    cfg, components = _build_components()
    model = components["model"]
    trainer = components["trainer"]
    dataset_template = components["dataset_template"]
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.0, -1.0], [-0.5, 0.25]]))
        model.linear.bias.copy_(torch.tensor([0.1, -0.2]))

    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = Path(tmpdir) / "tiny_expert"
        save_expert_package(
            package_path,
            expert_name="tiny",
            task="unit",
            model=model,
            trainer=trainer,
            dataset_template=dataset_template,
            builder={"module": __name__, "function": "build_tiny_components", "config": cfg},
            metadata={"experiment": "unit-test"},
        )

        loaded = load_expert_package(package_path, map_location="cpu")

        assert loaded.manifest.format_version == "sleepwalker-expert-v1"
        assert loaded.manifest.expert_name == "tiny"
        assert loaded.metadata["experiment"] == "unit-test"
        assert torch.allclose(loaded.model.linear.weight, model.linear.weight)
        assert torch.allclose(loaded.model.linear.bias, model.linear.bias)


def test_loaded_expert_runs_forward_on_compatible_batch():
    cfg, components = _build_components()
    model = components["model"]
    trainer = components["trainer"]
    dataset_template = components["dataset_template"]

    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = Path(tmpdir) / "tiny_expert"
        save_expert_package(
            package_path,
            expert_name="tiny",
            task="unit",
            model=model,
            trainer=trainer,
            dataset_template=dataset_template,
            builder={"module": __name__, "function": "build_tiny_components", "config": cfg},
        )

        loaded = load_expert_package(package_path, map_location="cpu")
        batch = {"data": torch.zeros(3, 4, 2)}
        outputs = loaded.forward(batch)

        assert outputs.shape == (3, 2)


def test_loaded_expert_freeze_and_unfreeze_toggle_requires_grad():
    cfg, components = _build_components()

    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = Path(tmpdir) / "tiny_expert"
        save_expert_package(
            package_path,
            expert_name="tiny",
            task="unit",
            model=components["model"],
            trainer=components["trainer"],
            dataset_template=components["dataset_template"],
            builder={"module": __name__, "function": "build_tiny_components", "config": cfg},
        )

        loaded = load_expert_package(package_path, map_location="cpu")
        loaded.freeze()
        assert all(not param.requires_grad for param in loaded.model.parameters())

        loaded.unfreeze()
        assert all(param.requires_grad for param in loaded.model.parameters())


def test_loaded_expert_restores_optimizer_and_scheduler_state():
    cfg, components = _build_components()
    model = components["model"]
    trainer = components["trainer"]
    optimizer = trainer.optimizer_fn(model)
    scheduler = trainer.lr_scheduler_fn(optimizer)

    batch = torch.ones(2, 4, 2)
    target = torch.tensor([0, 1])
    loss = torch.nn.functional.cross_entropy(model(batch), target)
    loss.backward()
    optimizer.step()
    scheduler.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = Path(tmpdir) / "tiny_expert"
        save_expert_package(
            package_path,
            expert_name="tiny",
            task="unit",
            model=model,
            trainer=trainer,
            dataset_template=components["dataset_template"],
            builder={"module": __name__, "function": "build_tiny_components", "config": cfg},
            optimizer=optimizer,
            scheduler=scheduler,
        )

        loaded = load_expert_package(package_path, map_location="cpu")
        restored_optimizer = loaded.build_optimizer()
        restored_scheduler = loaded.build_scheduler(restored_optimizer)

        original_state = optimizer.state_dict()
        restored_state = restored_optimizer.state_dict()
        assert restored_state["param_groups"] == original_state["param_groups"]
        assert restored_state["state"].keys() == original_state["state"].keys()
        for param_id in original_state["state"]:
            assert restored_state["state"][param_id].keys() == original_state["state"][param_id].keys()
            for key, value in original_state["state"][param_id].items():
                restored_value = restored_state["state"][param_id][key]
                if torch.is_tensor(value):
                    assert torch.allclose(restored_value, value)
                else:
                    assert restored_value == value
        assert restored_scheduler is not None
        assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_load_expert_package_restores_warmed_preprocessor_state():
    cfg, _ = _build_components()
    cfg = {**cfg, "robust_scaler": True, "scaler_channels": [0, 1]}
    components = build_tiny_components(cfg)
    model = components["model"]

    model.preprocessors[0].update(torch.randn(2, 4, 2))
    assert model.preprocessors[0].n.numel() == 2

    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = Path(tmpdir) / "tiny_expert"
        save_expert_package(
            package_path,
            expert_name="tiny",
            task="unit",
            model=model,
            trainer=components["trainer"],
            dataset_template=components["dataset_template"],
            builder={"module": __name__, "function": "build_tiny_components", "config": cfg},
        )

        loaded = load_expert_package(package_path, map_location="cpu")

        assert loaded.model.preprocessors[0].n.shape == model.preprocessors[0].n.shape
        assert torch.allclose(loaded.model.preprocessors[0].marker_heights, model.preprocessors[0].marker_heights)


def test_loaded_expert_rejects_dataset_contract_mismatch():
    cfg, components = _build_components()

    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = Path(tmpdir) / "tiny_expert"
        save_expert_package(
            package_path,
            expert_name="tiny",
            task="unit",
            model=components["model"],
            trainer=components["trainer"],
            dataset_template=components["dataset_template"],
            builder={"module": __name__, "function": "build_tiny_components", "config": cfg},
        )

        loaded = load_expert_package(package_path, map_location="cpu")
        bad_dataset = TinyDatasetTemplate(
            channels=["sig_a", "sig_c"],
            sample_frequency=1.0,
            total_input="4s",
            target_resolution="2s",
            stride="2s",
        )
        with pytest.raises(ValueError, match="Expected channels"):
            loaded.validate_dataset(bad_dataset)


def test_load_uses_recorded_builder_reference():
    cfg, components = _build_components()
    BUILDER_CALLS.clear()

    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = Path(tmpdir) / "tiny_expert"
        save_expert_package(
            package_path,
            expert_name="tiny",
            task="unit",
            model=components["model"],
            trainer=components["trainer"],
            dataset_template=components["dataset_template"],
            builder={"module": __name__, "function": "build_tiny_components", "config": cfg},
        )

        load_expert_package(package_path, map_location="cpu")

        assert BUILDER_CALLS == [cfg]


def test_package_writes_manifest_json():
    cfg, components = _build_components()

    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = Path(tmpdir) / "tiny_expert"
        save_expert_package(
            package_path,
            expert_name="tiny",
            task="unit",
            model=components["model"],
            trainer=components["trainer"],
            dataset_template=components["dataset_template"],
            builder={"module": __name__, "function": "build_tiny_components", "config": cfg},
            metadata={"tag": "check"},
        )

        manifest = json.loads((package_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["format_version"] == "sleepwalker-expert-v1"
        assert manifest["builder"]["module"] == __name__
        assert manifest["metadata"]["tag"] == "check"


def test_run_converges_on_expert_package_checkpoints():
    cfg, components = _build_components()
    trainer = components["trainer"]
    trainer.epochs = 1
    trainer.save_every = 1
    model = components["model"]
    dataset = TinyTrainDataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run(
            RunCfg(
                experiment_name="tiny-run",
                model_name="tiny-model",
                model=model,
                trainer=trainer,
                train_datasets=[dataset],
                val_datasets=[],
                test_datasets=[],
                batch_size=2,
                n_samples=None,
                num_workers_dataloader=0,
                collate_fn=lambda batch: {
                    "data": torch.stack([item["data"] for item in batch]),
                    "target": torch.stack([item["target"] for item in batch]),
                },
                log_path=tmpdir,
                expert_name="tiny",
                expert_task="unit",
                expert_builder={"module": __name__, "function": "build_tiny_components", "config": cfg},
            )
        )

        checkpoint_path = Path(result.train_result["checkpoint"])
        assert (checkpoint_path / "manifest.json").is_file()
        assert (checkpoint_path / "model_state.pt").is_file()

        loaded = load_expert_package(checkpoint_path, map_location="cpu")
        outputs = loaded.forward({"data": torch.zeros(1, 4, 2)})
        assert outputs.shape == (1, 2)


def test_train_arousal_expert_builder_reconstructs_components_without_data_access():
    components = build_arousal_expert_components(
        {
            "channels": ["eeg", "eog", "chin_emg", "ECG"],
            "clean": False,
            "grouped": True,
            "total_input": "30s",
            "model": "utime-small",
            "scaler": False,
            "arousal_weight": 1,
            "epochs": 2,
        }
    )

    assert components["trainer"].epochs == 2
    assert isinstance(components["dataset_template"], UnlabelledDataset)
    assert len(components["dataset_template"].get_input_channels()) == 4
    shape, meta = components["model"].input_spec()
    assert shape[2] == 4
    assert meta["n_channels"] == 4
    assert meta["layout"] == "BTC"
