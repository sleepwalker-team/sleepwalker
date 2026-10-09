import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from sleepwalker.datasets.normalizer import ConvertUnit
from sleepwalker.datasets.Basedataset import ChannelConfig, EDFFile
from sleepwalker.core.signal import unit_conversion_factor
from sleepwalker.datasets.HSP import HSP, get_channels as get_hsp_channels, correct_hsp_saturation
from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.datasets.normalizer import EEGFilterNormalizer, RecordingZScore
from sleepwalker.deployment import PackagedModel, save_packaged_model
from sleepwalker.deployment.predictions import format_prediction_batch
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel


DATA = Path(__file__).parent / "data"
SINGLE_CONTRACT = {"type": "single-head-multiclass", "classes": ["negative", "positive"], "sequence_len": 1, "target_resolution": "60s"}


class MeanClassifier(BaseModel, ClassifierModel):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.linear = torch.nn.Linear(1, n_classes)

    def compute(self, x):
        return self.linear(x.mean(dim=1)).unsqueeze(1)

    def input_spec(self):
        return (1, 600, 1), {"layout": "BTC", "ts_len": 600, "n_channels": 1}


def make_dataset(*, unit: str | None = "uV", correction=(), preprocessors=()):
    return UnlabelledDataset(
        channels=[ChannelConfig('EEG', ['EEG'], preprocessors=list(correction) + ([] if unit is None else [ConvertUnit(unit)]) + list(preprocessors))],
        sample_frequency=10,
        total_input="60s",
        stride="60s",
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
    dataset = HSP(channels=channels, sample_frequency=100, event_mapping={"desaturation": "desaturation"})

    expected = [ChannelConfig('SpO2', ['SaO2', 'SpO2', 'SPO2'], preprocessors={name: [ConvertUnit('%')] for name in ['SaO2', 'SpO2', 'SPO2']})]

    assert channels[0].logical_name == expected[0].logical_name

    assert channels[0].physical_names == expected[0].physical_names
    assert dataset.channels[0].preprocessors_for("SaO2")[0] is correct_hsp_saturation
    assert UnlabelledDataset.from_dataset(dataset).channels[0].preprocessors_for("SaO2")[0] is correct_hsp_saturation


def test_missing_unit_bypass_is_available_during_dataset_initialization(monkeypatch):
    original = EDFFile.from_edf

    def without_units(path):
        file = original(path)
        file.units["EEG"] = None
        return file

    monkeypatch.setattr(EDFFile, "from_edf", staticmethod(without_units))
    path = DATA / "signals_01.edf"
    unknown = make_dataset()
    unknown.initialize([path], num_workers=0, strict=True)
    assert unknown.get_item(unknown.edf_files[0], unknown.edf_files[0].start_date) is None
    assume_uv = lambda values, *, unit, is_recording: (values, "uV" if unit is None else unit)
    assumed_dataset = make_dataset(correction=[assume_uv])
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


def test_loading_legacy_package_moves_target_resolution_out_of_the_dataset(tmp_path):
    package = make_package()
    package.dataset.target_resolution = pd.Timedelta("20s")
    package.dataset._init_kwargs["target_resolution"] = "20s"
    package.classification_contract = dict(package.classification_contract)
    package.classification_contract.pop("target_resolution")

    loaded = PackagedModel.load(package.save(tmp_path / "legacy-package"))

    assert loaded.classification_contract["target_resolution"] == "0 days 00:00:20"
    assert loaded.classification_contract["target_offset"] == "0 days 00:00:20"
    assert not hasattr(loaded.dataset, "target_resolution")
    assert "target_resolution" not in loaded.dataset.dataset_kwargs()


def test_packaging_discards_label_pipeline(tmp_path):
    labelled = HSP(channels=[ChannelConfig('EEG', ['EEG'], preprocessors=[ConvertUnit('uV')])], sample_frequency=10, total_input="60s", event_mapping={"desaturation": "desaturation"})
    packaged = save_packaged_model(tmp_path / "package", name="tiny", task="unit-test", model=MeanClassifier(), classification_contract=SINGLE_CONTRACT, dataset=labelled)

    assert isinstance(packaged.dataset, UnlabelledDataset)
    assert packaged.dataset.event_mapping is None


def test_packaging_uses_first_component_of_multidataset(tmp_path):
    first = HSP(channels=[ChannelConfig('EEG', ['EEG'], preprocessors=[ConvertUnit('uV')])], sample_frequency=10, total_input="60s", event_mapping={"desaturation": "desaturation"})
    second = HSP(channels=[ChannelConfig('EEG', ['EEG'], preprocessors=[ConvertUnit('uV')])], sample_frequency=10, total_input="60s", event_mapping={"desaturation": "desaturation"})
    first.initialized = True
    second.initialized = True

    packaged = save_packaged_model(tmp_path / "package", name="tiny", task="unit-test", model=MeanClassifier(), classification_contract=SINGLE_CONTRACT, dataset=MultiDataset([first, second]))

    assert isinstance(packaged.dataset, UnlabelledDataset)
    assert packaged.dataset.get_input_channels() == first.get_input_channels()


def test_package_leaves_processor_comparability_to_the_caller():
    package = make_package()
    package.assert_compatible(make_dataset(unit="mV"))
    package.assert_compatible(make_dataset(preprocessors=[RecordingZScore()]))


def test_loaded_package_predicts_raw_edf(tmp_path):
    loaded = PackagedModel.load(make_package().save(tmp_path / "package"))
    predictions = loaded.predict_patient(DATA / "signals_01.edf", batch_size=64)

    assert not predictions.empty
    assert {"time", "prediction", "prob__negative", "prob__positive"}.issubset(predictions.columns)
    assert pd.to_datetime(predictions["time"]).is_monotonic_increasing


def test_dataset_prediction_uses_the_requested_rejection_strategy():
    dataset = make_dataset()
    dataset.initialize([DATA / "signals_01.edf"], num_workers=0, strict=True)

    predictions = make_package().predict_dataset(dataset, rejection_strategy="patient", progress=False)

    assert not predictions.empty
    assert dataset.rejection_strategy == "patient"


def test_dataset_prediction_reports_received_input_windows():
    dataset = make_dataset()
    patient = DATA / "signals_01.edf"
    dataset.initialize([patient], num_workers=0, strict=True)

    predictions, received = make_package().predict_dataset(dataset, progress=False, return_received_windows=True)

    assert not predictions.empty
    assert received == {str(patient): len(dataset)}


def test_sequence_predictions_use_target_start_timestamps():
    contract = {"type": "single-head-multiclass", "task": "event", "classes": ["negative", "positive"], "sequence_len": 2, "target_resolution": "20s", "target_offset": "30s"}
    start = pd.Timestamp("2024-01-01T00:00:20")
    frame = format_prediction_batch(contract, {"patient": ["patient"], "time": [start]}, torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]))
    assert frame["time"].tolist() == [start + pd.Timedelta(seconds=30), start + pd.Timedelta(seconds=40)]
    assert frame["task"].tolist() == ["event", "event"]
    assert frame["prediction_idx"].tolist() == [0, 1]
    assert frame["prediction"].tolist() == ["negative", "positive"]


def test_multitask_predictions_include_centered_task_offset():
    contract = {
        "type": "multitask",
        "tasks": {
            "sleep": {"classes": ["wake", "n2"], "n_steps": 1, "target_resolution": "30s", "target_offset": "5s"},
            "arousal": {"classes": ["no_arousal", "arousal"], "n_steps": 40, "target_resolution": "1s", "target_offset": "0s"},
        },
    }
    start = pd.Timestamp("2024-01-01T00:00:20")
    frame = format_prediction_batch(contract, {"patient": ["patient"], "time": [start]}, {"sleep": torch.zeros(1, 1, 2), "arousal": torch.zeros(1, 40, 2)})
    assert len(frame) == 41
    assert set(frame["task"]) == {"sleep", "arousal"}
    assert frame.loc[frame["task"] == "sleep", "time"].tolist() == [start + pd.Timedelta("5s")]
    assert frame.loc[frame["task"] == "arousal", "time"].iloc[-1] == start + pd.Timedelta("39s")


def test_conversion_rejection_survives_package_reload(monkeypatch, tmp_path):
    original = EDFFile.from_edf
    def conflicting_units(path):
        file = original(path)
        file.units["EEG"] = "%"
        return file

    monkeypatch.setattr(EDFFile, "from_edf", staticmethod(conflicting_units))
    package = PackagedModel.load(make_package().save(tmp_path / "package"))
    file = package.dataset.prepare_patient(DATA / "signals_01.edf", raise_errors=True)
    assert package.dataset.get_item(file, file.start_date) is None
