import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from sleepwalker.datasets.Basedataset import ChannelConfig, unit_conversion_factor
from sleepwalker.datasets.HSP import HSP, get_channels as get_hsp_channels
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.datasets.normalizer import EEGFilterNormalizer
from sleepwalker.deployment import Expert, save_expert_package
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer


DATA = Path(__file__).parent / "data"


class MeanClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)

    def forward(self, x):
        return self.linear(x.mean(dim=1))


def make_dataset(
    *,
    unit: str | None = "uV",
    assume_units_if_missing: bool = False,
    prepare_patient=None,
    normalizer=None,
):
    return UnlabelledDataset(
        channels=[ChannelConfig("EEG", ["EEG"], unit=unit, normalizer=normalizer)],
        sample_frequency=10,
        total_input="60s",
        target_resolution="60s",
        stride="60s",
        prepare_patient=prepare_patient,
        assume_units_if_missing=assume_units_if_missing,
    )


def make_expert(dataset=None):
    trainer = MulticlassTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        classes=["negative", "positive"],
        loss_function=torch.nn.functional.cross_entropy,
        device="cpu",
        warmup_device="cpu",
    )
    return Expert(
        name="tiny",
        task="unit-test",
        model=MeanClassifier(),
        dataset=make_dataset() if dataset is None else dataset,
        trainer=trainer,
        config={"seed": 7, "comment": "Synthetic round-trip test."},
    )


@pytest.mark.parametrize(
    ("source", "target", "factor"),
    [
        ("V", "uV", 1_000_000.0),
        ("mV", "uV", 1_000.0),
        ("µV", "uV", 1.0),
        ("%", "percent", 1.0),
    ],
)
def test_unit_conversion_factor(source, target, factor):
    assert unit_conversion_factor(source, target, assume_if_missing=False) == pytest.approx(factor)


def test_unit_conversion_rejects_missing_and_incompatible_units():
    with pytest.raises(ValueError, match="no unit metadata"):
        unit_conversion_factor("", "uV", assume_if_missing=False)
    assert unit_conversion_factor("", "uV", assume_if_missing=True) == 1.0
    with pytest.raises(ValueError, match="Incompatible"):
        unit_conversion_factor("%", "uV", assume_if_missing=False)


def test_hsp_header_correction_survives_unlabelled_clone():
    channels = get_hsp_channels(
        ["spo2"],
        grouped=True,
        normalize=False,
        sample_frequency=100,
    )
    dataset = HSP(
        channels=channels,
        sample_frequency=100,
        event_mapping={"desaturation": "desaturation"},
    )

    assert channels == [
        ChannelConfig("SpO2", ["SaO2", "SpO2", "SPO2"], normalizer=None, unit="%")
    ]
    assert {channel.unit for channel in channels} == {"%"}
    assert dataset.edf_unit_overrides["SaO2"] == "%"
    assert UnlabelledDataset.from_dataset(dataset).edf_unit_overrides["SaO2"] == "%"


def test_missing_unit_bypass_is_available_during_dataset_initialization(monkeypatch):
    import sleepwalker.datasets.Basedataset as basedataset_module

    original = basedataset_module.read_edf_meta

    def without_units(path):
        meta = original(path)
        meta["units"] = {**meta["units"], "EEG": ""}
        return meta

    monkeypatch.setattr(basedataset_module, "read_edf_meta", without_units)
    path = DATA / "signals_01.edf"

    strict_dataset = make_dataset(assume_units_if_missing=False)
    with pytest.raises(ValueError, match="no unit metadata"):
        strict_dataset.initialize([path], num_workers=0, strict=True)

    assumed_dataset = make_dataset(assume_units_if_missing=True)
    assumed_dataset.initialize([path], num_workers=0, strict=True)
    assert assumed_dataset.get_n_patients() == 1


def test_unlabelled_dataset_runs_prepare_patient():
    calls = []

    def prepare_patient(data_df, label_df, label_extra_df, patient=None):
        calls.append((label_df, label_extra_df, patient))
        return data_df, label_df, label_extra_df

    dataset = make_dataset(prepare_patient=prepare_patient)
    dataset.initialize([DATA / "signals_01.edf"], num_workers=0, strict=True)

    assert len(calls) == 1
    assert calls[0][0] is None
    assert calls[0][1] is None


def test_expert_roundtrip_preserves_objects_manifest_and_weights(tmp_path):
    expert = make_expert()
    with torch.no_grad():
        expert.model.linear.weight.copy_(torch.tensor([[1.0], [-0.5]]))
        expert.model.linear.bias.copy_(torch.tensor([0.1, -0.2]))

    path = expert.save(tmp_path / "expert")
    loaded = Expert.load(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["format_version"] == "sleepwalker-expert-v5"
    assert manifest["input_channels"] == ["EEG"]
    assert manifest["config"]["seed"] == 7
    assert isinstance(loaded.dataset, UnlabelledDataset)
    assert torch.allclose(loaded.model.linear.weight, expert.model.linear.weight)


def test_packaging_discards_label_pipeline_from_executable_expert(tmp_path):
    labelled = HSP(
        channels=[ChannelConfig("EEG", ["EEG"], unit="uV")],
        sample_frequency=10,
        total_input="60s",
        target_resolution="60s",
        event_mapping={"desaturation": "desaturation"},
    )
    source = make_expert(labelled)
    packaged = save_expert_package(
        tmp_path / "expert",
        expert_name=source.name,
        task=source.task,
        model=source.model,
        trainer=source.trainer,
        dataset=labelled,
        config=source.config,
    )

    assert isinstance(packaged.dataset, UnlabelledDataset)
    assert packaged.dataset.event_mapping is None
    manifest = json.loads((tmp_path / "expert" / "manifest.json").read_text())
    assert manifest["dataset_class"].endswith("UnlabelledDataset")


def test_expert_rejects_incompatible_preprocessing():
    expert = make_expert()
    incompatible = make_dataset(unit="mV")

    with pytest.raises(ValueError, match="Expected units"):
        expert.assert_compatible(incompatible)


def test_expert_rejects_changed_normalizer_configuration():
    expert = make_expert(make_dataset(normalizer=EEGFilterNormalizer(fs=100)))
    incompatible = make_dataset(
        normalizer=EEGFilterNormalizer(fs=100, lowcut=0.5)
    )

    with pytest.raises(ValueError, match="Normalizer configuration"):
        expert.assert_compatible(incompatible)


def test_loaded_expert_predicts_raw_edf(tmp_path):
    loaded = Expert.load(make_expert().save(tmp_path / "expert"))

    predictions = loaded.predict_edf(
        DATA / "signals_01.edf",
        batch_size=64,
    )

    assert not predictions.empty
    assert {"time", "prediction", "prob__negative", "prob__positive"}.issubset(
        predictions.columns
    )
    assert pd.to_datetime(predictions["time"]).is_monotonic_increasing


def test_prediction_missing_unit_override_does_not_hide_conflicts(monkeypatch, tmp_path):
    import sleepwalker.datasets.Basedataset as basedataset_module

    original = basedataset_module.read_edf_meta

    def conflicting_units(path):
        meta = original(path)
        meta["units"] = {**meta["units"], "EEG": "%"}
        return meta

    monkeypatch.setattr(basedataset_module, "read_edf_meta", conflicting_units)
    expert = Expert.load(make_expert().save(tmp_path / "expert"))

    with pytest.raises(ValueError, match="Incompatible"):
        expert.predict_edf(
            DATA / "signals_01.edf",
            assume_units_if_missing=True,
        )
