import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from sleepwalker.datasets.Basedataset import ChannelConfig, unit_conversion_factor
from sleepwalker.datasets.HSP import HSP, get_channels as get_hsp_channels
from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.datasets.normalizer import EEGFilterNormalizer
from sleepwalker.deployment import PackagedModel, save_packaged_model
from sleepwalker.deployment.predictions import format_prediction_batch
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel


DATA = Path(__file__).parent / "data"
SINGLE_CONTRACT = {"type": "single-head-multiclass", "classes": ["negative", "positive"], "sequence_len": 1}


class MeanClassifier(BaseModel, ClassifierModel):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.linear = torch.nn.Linear(1, n_classes)

    def compute(self, x):
        return self.linear(x.mean(dim=1)).unsqueeze(1)

    def input_spec(self):
        return (1, 600, 1), {"layout": "BTC", "ts_len": 600, "n_channels": 1}


def make_dataset(*, unit: str | None = "uV", assume_units_if_missing: bool = False, normalizer=None, z_normalize: bool = False):
    return UnlabelledDataset(
        channels=[ChannelConfig("EEG", ["EEG"], unit=unit, normalizer=normalizer)],
        sample_frequency=10,
        total_input="60s",
        target_resolution="60s",
        stride="60s",
        z_normalize=z_normalize,
        assume_units_if_missing=assume_units_if_missing,
    )


def make_package(dataset=None):
    return PackagedModel(
        name="tiny",
        task="unit-test",
        model=MeanClassifier(),
        dataset=make_dataset() if dataset is None else dataset,
        classification_contract=SINGLE_CONTRACT,
        config={"seed": 7, "comment": "Synthetic round-trip test."},
    )


@pytest.mark.parametrize(("source", "target", "factor"), [("V", "uV", 1_000_000.0), ("mV", "uV", 1_000.0), ("µV", "uV", 1.0), ("%", "percent", 1.0)])
def test_unit_conversion_factor(source, target, factor):
    assert unit_conversion_factor(source, target, assume_if_missing=False) == pytest.approx(factor)


def test_unit_conversion_rejects_missing_and_incompatible_units():
    with pytest.raises(ValueError, match="no unit metadata"):
        unit_conversion_factor("", "uV", assume_if_missing=False)
    assert unit_conversion_factor("", "uV", assume_if_missing=True) == 1.0
    with pytest.raises(ValueError, match="Incompatible"):
        unit_conversion_factor("%", "uV", assume_if_missing=False)


def test_online_retry_configuration_only_exposes_budget():
    dataset = make_dataset()

    assert dataset.online_max_tries == 128
    assert "online_retry_scope" not in dataset.dataset_kwargs()


def test_hsp_header_correction_survives_unlabelled_clone():
    channels = get_hsp_channels(["spo2"], grouped=True, normalize=False, sample_frequency=100)
    dataset = HSP(channels=channels, sample_frequency=100, event_mapping={"desaturation": "desaturation"}, z_normalize=True)

    assert channels == [ChannelConfig("SpO2", ["SaO2", "SpO2", "SPO2"], normalizer=None, unit="%")]
    assert dataset.edf_unit_overrides["SaO2"] == "%"
    assert UnlabelledDataset.from_dataset(dataset).edf_unit_overrides["SaO2"] == "%"
    assert UnlabelledDataset.from_dataset(dataset).z_normalize is True


def test_missing_unit_bypass_is_available_during_dataset_initialization(monkeypatch):
    import sleepwalker.datasets.Basedataset as basedataset_module

    original = basedataset_module.read_edf_meta

    def without_units(path):
        meta = original(path)
        meta["units"] = {**meta["units"], "EEG": ""}
        return meta

    monkeypatch.setattr(basedataset_module, "read_edf_meta", without_units)
    path = DATA / "signals_01.edf"
    with pytest.raises(ValueError, match="no unit metadata"):
        make_dataset().initialize([path], num_workers=0, strict=True)
    assumed_dataset = make_dataset(assume_units_if_missing=True)
    assumed_dataset.initialize([path], num_workers=0, strict=True)
    assert assumed_dataset.get_n_patients() == 1


def test_package_roundtrip_preserves_contract_manifest_and_weights(tmp_path):
    package = make_package()
    with torch.no_grad():
        package.model.linear.weight.copy_(torch.tensor([[1.0], [-0.5]]))
        package.model.linear.bias.copy_(torch.tensor([0.1, -0.2]))

    path = package.save(tmp_path / "package")
    loaded = PackagedModel.load(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["format_version"] == "sleepwalker-packaged-model-v1"
    assert manifest["input_channels"] == ["EEG"]
    assert manifest["classification"] == SINGLE_CONTRACT
    assert manifest["capabilities"] == ["classification"]
    assert isinstance(loaded.dataset, UnlabelledDataset)
    assert torch.allclose(loaded.model.linear.weight, package.model.linear.weight)


def test_packaging_discards_label_pipeline(tmp_path):
    labelled = HSP(channels=[ChannelConfig("EEG", ["EEG"], unit="uV")], sample_frequency=10, total_input="60s", target_resolution="60s", event_mapping={"desaturation": "desaturation"})
    packaged = save_packaged_model(tmp_path / "package", name="tiny", task="unit-test", model=MeanClassifier(), classification_contract=SINGLE_CONTRACT, dataset=labelled)

    assert isinstance(packaged.dataset, UnlabelledDataset)
    assert packaged.dataset.event_mapping is None


def test_packaging_uses_first_component_of_multidataset(tmp_path):
    first = HSP(channels=[ChannelConfig("EEG", ["EEG"], unit="uV")], sample_frequency=10, total_input="60s", target_resolution="60s", event_mapping={"desaturation": "desaturation"})
    second = HSP(channels=[ChannelConfig("EEG", ["EEG"], unit="uV")], sample_frequency=10, total_input="60s", target_resolution="60s", event_mapping={"desaturation": "desaturation"})
    first.initialized = True
    second.initialized = True

    packaged = save_packaged_model(tmp_path / "package", name="tiny", task="unit-test", model=MeanClassifier(), classification_contract=SINGLE_CONTRACT, dataset=MultiDataset([first, second]))

    assert isinstance(packaged.dataset, UnlabelledDataset)
    assert packaged.dataset.get_input_channels() == first.get_input_channels()


def test_package_rejects_incompatible_preprocessing():
    with pytest.raises(ValueError, match="Expected units"):
        make_package().assert_compatible(make_dataset(unit="mV"))


def test_package_rejects_changed_normalizer_configuration():
    package = make_package(make_dataset(normalizer=EEGFilterNormalizer(fs=100)))
    with pytest.raises(ValueError, match="Normalizer configuration"):
        package.assert_compatible(make_dataset(normalizer=EEGFilterNormalizer(fs=100, lowcut=0.5)))


def test_package_rejects_changed_recording_z_normalization():
    with pytest.raises(ValueError, match="z_normalize"):
        make_package(make_dataset(z_normalize=True)).assert_compatible(make_dataset())


def test_loaded_package_predicts_raw_edf(tmp_path):
    loaded = PackagedModel.load(make_package().save(tmp_path / "package"))
    predictions = loaded.predict_edf(DATA / "signals_01.edf", batch_size=64)

    assert not predictions.empty
    assert {"time", "prediction", "prob__negative", "prob__positive"}.issubset(predictions.columns)
    assert pd.to_datetime(predictions["time"]).is_monotonic_increasing


def test_sequence_predictions_use_target_start_timestamps():
    contract = {"type": "single-head-multiclass", "classes": ["negative", "positive"], "sequence_len": 2}
    start = pd.Timestamp("2024-01-01T00:00:20")
    frame = format_prediction_batch(contract, {"patient": ["patient"], "time": [start]}, torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]), target_resolution="20s")
    assert frame["time"].tolist() == [start, start + pd.Timedelta(seconds=10)]


def test_multitask_predictions_include_centered_task_offset():
    contract = {
        "type": "multitask",
        "tasks": {
            "sleep": {"classes": ["wake", "n2"], "n_steps": 1, "target_resolution": "30s", "target_offset": "5s"},
            "arousal": {"classes": ["no_arousal", "arousal"], "n_steps": 40, "target_resolution": "1s", "target_offset": "0s"},
        },
    }
    start = pd.Timestamp("2024-01-01T00:00:20")
    frame = format_prediction_batch(contract, {"patient": ["patient"], "time": [start]}, {"sleep": torch.zeros(1, 1, 2), "arousal": torch.zeros(1, 40, 2)}, target_resolution="40s")
    assert frame.loc[0, "sleep__time"] == start + pd.Timedelta("5s")
    assert frame.loc[0, "arousal__step_39__time"] == start + pd.Timedelta("39s")


def test_prediction_missing_unit_override_does_not_hide_conflicts(monkeypatch, tmp_path):
    import sleepwalker.datasets.Basedataset as basedataset_module

    original = basedataset_module.read_edf_meta

    def conflicting_units(path):
        meta = original(path)
        meta["units"] = {**meta["units"], "EEG": "%"}
        return meta

    monkeypatch.setattr(basedataset_module, "read_edf_meta", conflicting_units)
    package = PackagedModel.load(make_package().save(tmp_path / "package"))
    with pytest.raises(ValueError, match="Incompatible"):
        package.predict_edf(DATA / "signals_01.edf", assume_units_if_missing=True)
