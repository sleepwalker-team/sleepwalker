"""EDF decoding, independent fitted state and recording/window processing."""

from datetime import datetime
from unittest.mock import Mock

import cloudpickle
import numpy as np
import pandas as pd
import pyedflib
import pytest
import sleepwalker.core.signal as signal
import torch

from sleepwalker.datasets.normalizer import ConvertUnit
from sleepwalker.config import build_channel
from sleepwalker.core.signal import edf_calibration, read_edf_meta, read_edf_native
from sleepwalker.datasets.Apples import Apples, get_preprocessors as apples_preprocessors, correct_apples_eeg, correct_apples_waveform, correct_apples_saturation
from sleepwalker.datasets.Basedataset import BaseDataset, ChannelConfig, EDFFile
from sleepwalker.core.signal import unit_conversion_factor
from sleepwalker.datasets.EDFCache import EDFCache
from sleepwalker.datasets.PairedDataset import PairedDataset
from sleepwalker.datasets.Ruhrlandklinik import get_preprocessors as ruhrland_preprocessors
from sleepwalker.datasets.SHHS import correct_shhs_waveform, correct_shhs_saturation
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.datasets.normalizer import FixedScale, RecordingRobustScale, RecordingZScore, RespirationFilterNormalizer, SaturationFilterNormalizer
from sleepwalker.deployment.package import assert_single_dataset_compatible
from sleepwalker.deployment import PackagedModel
from sleepwalker.models.BaseModel import BaseModel, ClassifierModel


def write_edf(path, specifications, frequency=20, seconds=20):
    headers, signals = [], []
    time = np.arange(frequency * seconds) / frequency
    for specification in specifications:
        specification = dict(specification)
        samples = specification.pop("samples", None)
        header = dict(label="A", dimension="uV", physical_min=-100, physical_max=100, digital_min=-32768, digital_max=32767, transducer="", prefilter="", sample_frequency=frequency)
        header.update(specification)
        signals.append(np.round(1500 * np.sin(time * 2.1) + 900 * np.cos(time * .7) + 120).astype(np.int32) if samples is None else np.full(len(time), samples, dtype=np.int32))
        headers.append(header)
    with pyedflib.EdfWriter(str(path), len(headers)) as writer:
        writer.setStartdatetime(datetime(2024, 1, 1))
        writer.setSignalHeaders(headers)
        writer.writeSamples(signals, digital=True)
    return path


@pytest.fixture
def recording(tmp_path):
    return write_edf(tmp_path / "recording.edf", [{"label": "A"}, {"label": "B", "dimension": "cmH2O", "physical_min": 10, "physical_max": -10}])


def dataset(channels, **kwargs):
    result = UnlabelledDataset(channels=channels, sample_frequency=40, total_input="5s", stride="1s", **kwargs)
    return result


def filtered_z():
    return [RespirationFilterNormalizer(fs=40, normalize=False), RecordingZScore()]


@pytest.mark.parametrize("unit,target,factor", [("V", "uV", 1e6), ("cmH2O", "Pa", 98.0665), ("L/min", "L/s", 1/60), (".", "relative", 1), ("counts", "counts", 1)])
def test_unit_families(unit, target, factor):
    assert unit_conversion_factor(unit, target, assume_if_missing=False) == factor


@pytest.mark.parametrize("source,target", [("cmH2O", "uV"), ("counts", "uV"), ("relative", "uV"), ("L/m", "L/min")])
def test_conversion_does_not_infer_calibration(source, target):
    with pytest.raises(ValueError):
        unit_conversion_factor(source, target, assume_if_missing=False)


@pytest.mark.parametrize("name", ["read_mode", "preprocessors"])
def test_alias_mappings_require_complete_coverage(name):
    with pytest.raises(ValueError, match="cover exactly"):
        ChannelConfig("group", ["A", "B"], **{name: {"A": [] if name == "preprocessors" else "physical"}})


def test_python_and_yaml_construct_identical_processing():
    actual = build_channel({'logical_name': 'signal', 'physical_names': ['A'], 'read_mode': 'digital', 'preprocessors': [{'name': 'sleepwalker.datasets.normalizer.ConvertUnit.ConvertUnit', 'target': 'counts'}, {'name': 'sleepwalker.datasets.normalizer.RecordingZScore'}]})
    expected = ChannelConfig('signal', ['A'], read_mode='digital', preprocessors=[ConvertUnit('counts'), RecordingZScore()])
    assert actual.read_mode_for("A") == expected.read_mode_for("A")
    assert vars(actual.preprocessors_for("A")[0]) == vars(expected.preprocessors_for("A")[0])


@pytest.mark.parametrize("name", ["A", "B"])
def test_physical_and_digital_scaling_agree_including_negative_gain(recording, name):
    physical = dataset([ChannelConfig('signal', [name], preprocessors=[ConvertUnit('uV' if name == 'A' else 'Pa'), *filtered_z()])])
    digital = dataset([ChannelConfig('signal', [name], read_mode='digital', preprocessors=[ConvertUnit('counts'), *filtered_z()])])
    first, second = (data.prepare_patient(recording, raise_errors=True) for data in (physical, digital))
    start = first.start_date + pd.Timedelta("4s")
    np.testing.assert_allclose(physical.get_item(first, start)["data"], digital.get_item(second, start)["data"], atol=1e-6)


def test_fit_is_independent_per_alias_and_never_runs_in_windows(recording, monkeypatch):
    prototype = RecordingZScore()
    data = dataset([ChannelConfig('group', ['A', 'B'], preprocessors={'A': [ConvertUnit('uV'), prototype], 'B': [ConvertUnit('Pa'), prototype]})])
    file = data.prepare_patient(recording, raise_errors=True)
    a, b = (file.preprocessors[("group", name)][-1] for name in ("A", "B"))
    assert prototype.mean is None and a is not b and a.mean != b.mean
    original = (a.mean, a.std, b.mean, b.std)
    for step in (a, b):
        monkeypatch.setattr(step, "fit", Mock(side_effect=AssertionError("window refitted")))
    for offset in (0, 2, 5):
        data.get_item(file, file.start_date + pd.Timedelta(seconds=offset))
    assert original == (a.mean, a.std, b.mean, b.std)


def test_shared_source_has_independent_units_and_scalers(recording):
    data = dataset([ChannelConfig('fixed', ['A'], preprocessors=[ConvertUnit('mV'), FixedScale(std=2)]), ChannelConfig('z', ['A'], preprocessors=[ConvertUnit('uV'), RecordingZScore()]), ChannelConfig('robust', ['A'], preprocessors=[ConvertUnit('uV'), RecordingRobustScale()])])
    file = data.prepare_patient(recording, raise_errors=True)
    values = data.get_item(file, file.start_date)["data"]
    assert values.shape == (200, 3)
    assert not np.allclose(values[:, 0], values[:, 1])
    assert not np.allclose(values[:, 1], values[:, 2])


@pytest.mark.parametrize("resample_type", ["nearest", "polyphase"])
def test_recording_scaler_is_reused_for_overlapping_windows(recording, resample_type):
    data = dataset([ChannelConfig("signal", ["A"], preprocessors=[RecordingZScore()])], resample_type=resample_type)
    file = data.prepare_patient(recording, raise_errors=True)
    first = data.get_item(file, file.start_date + pd.Timedelta("2s"))["data"]
    second = data.get_item(file, file.start_date + pd.Timedelta("4s"))["data"]
    scaler = file.preprocessors[("signal", "A")][0]
    for offset, transformed in ((2, first), (4, second)):
        start = file.start_date + pd.Timedelta(seconds=offset)
        raw = file.get_x(start, start + data.total_input, data.sample_frequency, resample_type)
        raw = data.ensure_timeseries_length(raw, start)
        np.testing.assert_allclose(transformed[:, 0], scaler(raw[["A"]].to_numpy(), unit="uV", is_recording=False)[0][:, 0], atol=1e-6)



@pytest.mark.parametrize("resample_type", ["nearest", "polyphase"])
def test_cache_preserves_boundaries_and_distinguishes_modes(recording, tmp_path, resample_type):
    data = dataset([ChannelConfig('signal', ['A'], preprocessors=[ConvertUnit('uV'), *filtered_z()])], resample_type=resample_type)
    file = data.prepare_patient(recording, raise_errors=True)
    start = file.start_date + pd.Timedelta("4s")
    expected = data.get_item(file, start)["data"]
    with EDFCache(max_patients=2, directory=tmp_path / "cache") as cache:
        data.set_edf_cache(cache)
        np.testing.assert_allclose(data.get_item(file, start)["data"], expected, atol=1e-6)
        digital = dataset([ChannelConfig('signal', ['A'], read_mode='digital', preprocessors=[ConvertUnit('counts'), *filtered_z()])], resample_type=resample_type)
        digital_file = digital.prepare_patient(recording, raise_errors=True)
        digital_expected = digital.get_item(digital_file, start)["data"]
        digital.set_edf_cache(cache)
        np.testing.assert_allclose(digital.get_item(digital_file, start)["data"], digital_expected, atol=1e-6)
        data.set_edf_cache(None)


class WindowCenter:

    def __call__(self, values, *, unit, is_recording):
        return values - values.mean(), unit


def test_transform_only_step_receives_the_actual_window(recording):
    data = dataset([ChannelConfig("signal", ["A"], preprocessors=[RecordingZScore(), WindowCenter()])])
    file = data.prepare_patient(recording, raise_errors=True)
    actual = data.get_item(file, file.start_date)["data"]
    assert abs(float(actual[:-1].mean())) < 1e-6
    assert actual[-1] == actual[-2]


def test_transform_only_step_can_precede_recording_fit(recording):
    data = dataset([ChannelConfig("signal", ["A"], preprocessors=[WindowCenter(), RecordingZScore()])])
    file = data.prepare_patient(recording, raise_errors=True)
    scaler = file.preprocessors[("signal", "A")][1]
    assert abs(scaler.mean) < 1e-12
    start = file.start_date + pd.Timedelta("3s")
    raw = file.get_x(start, start + data.total_input, data.sample_frequency, data.resample_type)
    raw = raw[["A"]].to_numpy()
    expected = scaler(WindowCenter()(raw, unit="uV", is_recording=False)[0], unit="uV", is_recording=False)[0]
    actual = data.get_item(file, start)["data"]
    np.testing.assert_allclose(actual[:len(expected)], expected, atol=1e-6)
    np.testing.assert_allclose(actual[len(expected):], expected[-1:], atol=1e-6)


def test_read_mode_can_differ_between_aliases(recording):
    cfg = ChannelConfig('signal', ['A', 'B'], read_mode={'A': 'physical', 'B': 'digital'}, preprocessors={'A': [ConvertUnit('uV'), RecordingZScore()], 'B': [ConvertUnit('counts'), RecordingZScore()]})
    data = dataset([cfg], group_sampling_strategy="none")
    file = data.prepare_patient(recording, raise_errors=True)
    start = file.start_date + pd.Timedelta("3s")
    actual = data.get_item(file, start)["data"]
    for index, name in enumerate(cfg.physical_names):
        raw = file.get_x(start, start + data.total_input, data.sample_frequency, data.resample_type, channels=[name], read_mode=cfg.read_mode_for(name))
        raw = data.ensure_timeseries_length(raw, start).to_numpy()
        expected = file.preprocessors[("signal", name)][-1](raw, unit="uV" if name == "A" else "counts", is_recording=False)[0]
        np.testing.assert_allclose(actual[:, index], expected[:, 0], atol=1e-6)


class AnnotatedDataset(BaseDataset):
    def get_event_df(self, edf_path, start_datetime):
        return pd.DataFrame({"Label": ["keep", "drop"], "Starttime": [start_datetime, start_datetime + pd.Timedelta("10s")], "Endtime": [start_datetime + pd.Timedelta("10s"), start_datetime + pd.Timedelta("20s")]})


def test_annotation_callback_does_not_change_recording_fitting_interval(recording):
    means = []
    for label in ("keep", "drop"):
        data = AnnotatedDataset(channels=[ChannelConfig("signal", ["A"], preprocessors=[RecordingZScore()])], sample_frequency=40, total_input="5s", stride="1s", event_mapping={"keep": "keep", "drop": "drop"}, prepare_patient=lambda label_df, label_extra_df, patient: (label_df[label_df.Label == label], label_extra_df))
        file = data.prepare_patient(recording, raise_errors=True)
        means.append(file.preprocessors[("signal", "A")][0].mean)
    assert means[0] == means[1]


def test_invalid_unused_alias_keeps_valid_source(recording):
    data = dataset([ChannelConfig('signal', ['B', 'A'], preprocessors=[ConvertUnit('uV'), RecordingZScore()])], group_sampling_strategy="first")
    file = data.prepare_patient(recording, raise_errors=True)
    assert data.select_channels(file.channels, file.preprocessors)[0][1] == "A"


@pytest.mark.parametrize("filtered", [False, True])
def test_flat_unused_alias_keeps_valid_source(tmp_path, filtered):
    path = write_edf(tmp_path / "flat.edf", [{"label": "flat", "samples": 10}, {"label": "A"}])
    data = dataset([ChannelConfig("signal", ["flat", "A"], preprocessors=filtered_z() if filtered else [RecordingZScore()])], group_sampling_strategy="first")
    file = data.prepare_patient(path, raise_errors=True)
    assert data.select_channels(file.channels, file.preprocessors)[0][1] == "A"


def test_percent_filter_rejects_counts_or_already_normalized_values(recording):
    for steps in ([SaturationFilterNormalizer(fs=40, normalize=False)], [RecordingZScore(), SaturationFilterNormalizer(fs=40, normalize=False)]):
        data = dataset([ChannelConfig("saturation", ["A"], read_mode="digital", preprocessors=steps)])
        file = data.prepare_patient(recording, raise_errors=True)
        assert data.get_item(file, file.start_date) is None


def test_percent_acceptance_is_an_explicit_preprocessor(tmp_path):
    path = write_edf(tmp_path / "bad.edf", [{"label": "SpO2", "dimension": "%", "physical_min": 0, "physical_max": 255, "digital_min": 0, "digital_max": 255, "samples": 255}])
    def reject_invalid_percent(values, *, unit, is_recording):
        return None if np.mean((values < 0) | (values > 100)) > .05 else (values, unit)
    data = dataset([ChannelConfig("saturation", ["SpO2"], preprocessors=[reject_invalid_percent, SaturationFilterNormalizer(fs=40, normalize=False)])])
    file = data.prepare_patient(path, raise_errors=True)
    assert data.get_item(file, file.start_date) is None
    # No universal threshold: omitting the caller's check permits clipping.
    data.channels[0].preprocessors = [SaturationFilterNormalizer(fs=40, normalize=False)]
    file = data.prepare_patient(path, raise_errors=True)
    np.testing.assert_allclose(data.get_item(file, file.start_date)["data"], 100)


@pytest.mark.parametrize("normalizer", [RecordingZScore(), RecordingRobustScale()])
def test_flat_channels_are_not_amplified(normalizer):
    assert normalizer.fit(np.full((100, 1), 127.0), unit="uV", is_recording=True) is None


@pytest.mark.parametrize(("processor", "expected"), [(correct_shhs_waveform, "relative"), (correct_shhs_saturation, "%"), (correct_apples_eeg, "uV"), (correct_apples_waveform, "relative"), (correct_apples_saturation, "%")])
def test_unit_corrections_only_supply_missing_labels(processor, expected):
    values = np.array([[93.5], [95.2]])
    for unit in (None, "counts", "uV", "%"):
        actual, output_unit = processor(values, unit=unit, is_recording=False)
        assert actual is values
        assert output_unit == (expected if unit is None else unit)


def test_corrections_survive_inference_cloning_and_reload(tmp_path):
    path = write_edf(tmp_path / "apples.edf", [{"label": "LOC", "dimension": "."}])
    source = Apples(channels=[ChannelConfig('eog', ['LOC'], preprocessors=[*apples_preprocessors('LOC'), ConvertUnit('uV'), RecordingZScore()])], sample_frequency=40, total_input="5s")
    for inference in (source, UnlabelledDataset.from_dataset(source), UnlabelledDataset.from_dataset(source).clone()):
        loaded = cloudpickle.loads(cloudpickle.dumps(inference))
        file = loaded.prepare_patient(path, raise_errors=True)
        assert file.units["LOC"] is None
        assert loaded.get_item(file, file.start_date)["data"].shape == (200, 1)


def test_spo2_decoding_retains_nonidentity_mapping_and_absolute_percent(tmp_path):
    path = write_edf(tmp_path / "spo2.edf", [{"label": "SpO2", "dimension": ".", "physical_min": -13, "physical_max": 115, "digital_min": 0, "digital_max": 255, "samples": 205}])
    source = Apples(channels=[ChannelConfig('saturation', ['SpO2'], preprocessors=[*apples_preprocessors('SpO2'), ConvertUnit('%'), SaturationFilterNormalizer(fs=40, normalize=False)])], sample_frequency=40, total_input="5s")
    file = source.prepare_patient(path, raise_errors=True)
    expected = 205 * 128/255 - 13
    np.testing.assert_allclose(source.get_item(file, file.start_date)["data"], expected, atol=1e-4)


def test_ruhrland_unit_recipes_reject_incompatible_saturation():
    values = np.full((100, 1), 32767.)
    steps = ruhrland_preprocessors("Saturation", None, True, None, 40)
    assert steps[0](values, unit="uV", is_recording=False) is None
    values = np.arange(100.).reshape(-1, 1)
    assert ConvertUnit("cmH2O")(values, unit="cmH2O", is_recording=False)[1] == "cmH2O"
    assert ConvertUnit("V/s")(values, unit="V/s", is_recording=False)[1] == "V/s"


@pytest.mark.parametrize("mode", ["physical", "digital"])
def test_native_reader_never_falls_back_to_physical_volts(monkeypatch, mode):
    monkeypatch.setattr(signal.pyedflib, "EdfReader", Mock(side_effect=RuntimeError("bad EDF")))
    fallback = Mock(side_effect=AssertionError("MNE substituted volts"))
    monkeypatch.setattr(signal.mne.io, "read_raw_edf", fallback)
    with pytest.raises(ValueError, match="Controlled EDF decoding"):
        read_edf_native("bad.edf", ["A"], None, None, read_mode=mode)
    fallback.assert_not_called()


def test_package_checks_geometry_without_inferred_processor_contracts():
    a = dataset([ChannelConfig("signal", ["A"], preprocessors=[RecordingZScore(), FixedScale(std=2)])])
    b = dataset([ChannelConfig("signal", ["A"], preprocessors=[FixedScale(std=2), RecordingZScore()])])
    assert_single_dataset_compatible(a, b)
    b.sample_frequency = 50
    with pytest.raises(ValueError, match="sample_frequency"):
        assert_single_dataset_compatible(a, b)


def test_missing_unit_requires_explicit_correction_for_physical_conversion(tmp_path):
    path = write_edf(tmp_path / "unit.edf", [{"label": "A", "dimension": "."}])
    data = dataset([ChannelConfig("signal", ["A"], preprocessors=[ConvertUnit("uV")])])
    file = data.prepare_patient(path, raise_errors=True)
    assert file.units["A"] is None
    assert data.get_item(file, file.start_date) is None
    assume_uv = lambda values, *, unit, is_recording: (values, "uV" if unit is None else unit)
    data = dataset([ChannelConfig("signal", ["A"], preprocessors=[assume_uv, ConvertUnit("uV")])])
    file = data.prepare_patient(path, raise_errors=True)
    assert file.units["A"] is None
    assert data.get_item(file, file.start_date) is not None
    assert not hasattr(file, "metadata") and not hasattr(file, "processing_audit")


def test_final_channel_callback_creates_references_after_conversion_and_survives_cloning(tmp_path):
    path = write_edf(tmp_path / "refs.edf", [{"label": "A", "dimension": "V", "physical_min": -.001, "physical_max": .001}, {"label": "B", "physical_min": -200, "physical_max": 200}])
    calls = []

    def reference(channels):
        assert set(channels) == {"source_a", "source_b"}
        a, a_unit = channels["source_a"]
        b, b_unit = channels["source_b"]
        assert a_unit == b_unit == "uV"
        calls.append(len(a))
        return {"forward": (a - b, "uV"), "reverse": (b - a, "uV")}

    data = dataset([ChannelConfig('source_a', ['A'], preprocessors=[ConvertUnit('uV')]), ChannelConfig('source_b', ['B'], preprocessors=[ConvertUnit('uV')])], prepare_channels=reference, input_channels=["forward", "reverse"])
    for source in (data, data.clone()):
        file = source.prepare_patient(path, raise_errors=True)
        assert file.channels == ["A", "B"]
        actual = source.get_item(file, file.start_date)["data"].numpy()
        np.testing.assert_allclose(actual[:, 0], -actual[:, 1], atol=1e-6)
        assert source.get_input_channels() == ["forward", "reverse"]
    assert calls == [199, 199]
    digital = dataset([ChannelConfig('source_a', ['A'], read_mode='digital', preprocessors=[ConvertUnit('uV')]), ChannelConfig('source_b', ['B'], read_mode='digital', preprocessors=[ConvertUnit('uV')])], prepare_channels=reference, input_channels=["forward", "reverse"])
    file = digital.prepare_patient(path, raise_errors=True)
    assert digital.get_item(file, file.start_date) is None


class ProcessingClassifier(BaseModel, ClassifierModel):
    def compute(self, values):
        mean = values.mean(dim=1)
        return torch.stack((mean, -mean), dim=-1)

    def input_spec(self):
        return (1, 200, 1), {"layout": "BTC"}


def test_new_processing_package_predictions_survive_save_reload(tmp_path):
    path = write_edf(tmp_path / "apples.edf", [{"label": "LOC", "dimension": "."}])
    source = Apples(channels=[ChannelConfig('EOG', ['LOC'], preprocessors=[*apples_preprocessors('LOC'), ConvertUnit('uV'), RecordingZScore()])], sample_frequency=40, total_input="5s", stride="5s")
    contract = {"type": "single-head-multiclass", "classes": ["positive", "negative"], "sequence_len": 1, "target_resolution": "5s"}
    package = PackagedModel(name="processing", model=ProcessingClassifier(), dataset=UnlabelledDataset.from_dataset(source), task="test", classification_contract=contract)
    before = package.predict_patient(path, progress=False)
    loaded = PackagedModel.load(package.save(tmp_path / "package"))
    after = loaded.predict_patient(path, progress=False)
    pd.testing.assert_frame_equal(before, after)
    assert not after.empty
    assert loaded.dataset.channels[0].preprocessors[-1].mean is None


def test_units_and_scope_follow_ordered_prefix_and_multiple_fits(recording):
    calls = []

    class Observe:
        def __init__(self, expected_unit):
            self.expected_unit = expected_unit

        def fit(self, values, *, unit, is_recording):
            assert is_recording and unit == self.expected_unit and values.shape[1] == 1 and len(values) >= 799
            self.mean = values.mean()
            return self

        def __call__(self, values, *, unit, is_recording):
            assert unit == self.expected_unit
            return values, unit

    def observe(values, *, unit, is_recording):
        calls.append((unit, is_recording, len(values)))
        return values, unit

    data = dataset([ChannelConfig("signal", ["A"], preprocessors=[ConvertUnit("mV"), Observe("mV"), RecordingZScore(), observe, Observe("dimensionless")])])
    file = data.prepare_patient(recording, raise_errors=True)
    assert calls == [("dimensionless", True, 799)]
    assert abs(file.preprocessors[("signal", "A")][-1].mean) < 1e-12
    data.get_item(file, file.start_date)
    assert calls[-1] == ("dimensionless", False, 199)
    assert file.units["A"] == "uV"


def test_fit_rejection_tries_aliases_and_can_exclude_a_recording(recording):
    class RejectFit:
        def fit(self, values, *, unit, is_recording):
            assert is_recording
            return None

        def __call__(self, values, *, unit, is_recording):
            raise AssertionError("Rejected alias was applied")

    data = dataset([ChannelConfig("signal", ["A", "B"], preprocessors={"A": [RejectFit()], "B": [RecordingZScore()]})])
    file = data.prepare_patient(recording, raise_errors=True)
    assert ("signal", "A") not in file.preprocessors
    assert data.get_item(file, file.start_date) is not None
    data.channels[0].preprocessors = [RejectFit()]
    assert data.prepare_patient(recording) is None
    with pytest.raises(ValueError, match="No eligible"):
        data.prepare_patient(recording, raise_errors=True)


def test_prefix_rejection_stops_fit_and_window_check_preserves_recording(recording):
    class NeverFit:
        def fit(self, values, *, unit, is_recording):
            raise AssertionError("Prefix rejection was ignored")

        def __call__(self, values, *, unit, is_recording):
            raise AssertionError("Unfitted alias was applied")

    def reject(values, *, unit, is_recording):
        return None

    def windows_only(values, *, unit, is_recording):
        return (values, unit) if is_recording else None

    data = dataset([ChannelConfig("signal", ["A", "B"], preprocessors={"A": [reject, NeverFit()], "B": [windows_only, RecordingZScore()]})])
    file = data.prepare_patient(recording, raise_errors=True)
    scaler = file.preprocessors[("signal", "B")][1]
    state = vars(scaler).copy()
    assert data.get_item(file, file.start_date) is None
    assert vars(scaler) == state and ("signal", "B") in file.preprocessors


@pytest.mark.parametrize("strategy", ["first", "random", "none"])
def test_window_alias_rejection_and_fallback(recording, strategy):
    def reject(values, *, unit, is_recording):
        return None

    data = dataset([ChannelConfig("signal", ["A", "B"], read_mode={"A": "physical", "B": "digital"}, preprocessors={"A": [reject], "B": [FixedScale(std=10)]})], group_sampling_strategy=strategy)
    file = data.prepare_patient(recording, raise_errors=True)
    before = list(file.preprocessors)
    item = data.get_item(file, file.start_date)
    if strategy == "none":
        assert item is None
    else:
        raw = file.get_x(file.start_date, file.start_date + data.total_input, 40, "nearest", channels=["B"], read_mode="digital")
        np.testing.assert_allclose(item["data"][:, 0], data.ensure_timeseries_length(raw, file.start_date)["B"] / 10)
    assert list(file.preprocessors) == before


@pytest.mark.parametrize("where", ["fit", "prefix", "window"])
def test_processor_errors_are_visible_during_initialization_and_access(recording, where):
    class BrokenFit:
        def fit(self, values, *, unit, is_recording):
            raise ValueError("processor bug")

        def __call__(self, values, *, unit, is_recording):
            return values, unit

    def broken(values, *, unit, is_recording):
        raise ValueError("processor bug")

    steps = {"fit": [BrokenFit()], "prefix": [broken, RecordingZScore()], "window": [broken]}[where]
    data = dataset([ChannelConfig("signal", ["A"], preprocessors=steps)])
    with pytest.raises(ValueError, match="processor bug"):
        data.initialize([recording], num_workers=0)
        data.get_item(data.edf_files[0], data.edf_files[0].start_date)


def test_functions_lambdas_repairs_and_scope_survive_serialization(recording):
    def repair(values, *, unit, is_recording):
        repaired = values.copy()
        repaired[~np.isfinite(repaired)] = 0
        return repaired, unit

    identity = lambda values, *, unit, is_recording: (values, unit)
    data = dataset([ChannelConfig("signal", ["A"], preprocessors=[repair, identity, RecordingZScore()])])
    loaded = cloudpickle.loads(cloudpickle.dumps(data))
    file = loaded.prepare_patient(recording, raise_errors=True)
    assert loaded.get_item(file, file.start_date) is not None
    values, unit = repair(np.array([[np.nan], [3.]]), unit="uV", is_recording=False)
    np.testing.assert_array_equal(values, [[0], [3]])
    assert unit == "uV"


def test_invalid_conversion_target_is_a_configuration_error():
    with pytest.raises(ValueError, match="Unsupported target"):
        ConvertUnit("microvolts_typo")


@pytest.mark.parametrize("method", [None, "forkserver"])
def test_local_processors_survive_parallel_preparation(recording, method):
    class LocalScaler(RecordingZScore):
        pass

    identity = lambda values, *, unit, is_recording: (values, unit)
    data = dataset([ChannelConfig("signal", ["A"], preprocessors=[identity, LocalScaler()])])
    data.initialize([recording], num_workers=2, strict=True, multiprocessing_start_method=method)
    file = data.edf_files[0]
    assert data.get_item(file, file.start_date) is not None
    assert file.preprocessors[("signal", "A")][-1].std > 0
    assert data.channels[0].preprocessors[-1].std is None


def test_sparse_annotation_gaps_do_not_change_loaded_data_or_fitting(tmp_path):
    path = write_edf(tmp_path / "sparse.edf", [{"label": "A"}], seconds=60)

    class SparseDataset(BaseDataset):
        def get_event_df(self, edf_path, start_datetime):
            return pd.DataFrame({"Label": ["keep", "keep"], "Starttime": [start_datetime, start_datetime + pd.Timedelta("55s")], "Endtime": [start_datetime + pd.Timedelta("5s"), start_datetime + pd.Timedelta("60s")]})

    data = SparseDataset(channels=[ChannelConfig("signal", ["A"], preprocessors=[RecordingZScore()])], sample_frequency=40, total_input="5s", stride="1s", event_mapping={"keep": "keep"})
    file = data.prepare_patient(path, raise_errors=True)
    raw = file.get_x(file.start_date, file.end_date, 40, "nearest")
    scaler = file.preprocessors[("signal", "A")][0]
    assert scaler.mean == pytest.approx(raw.A.mean())
    assert scaler.std == pytest.approx(raw.A.std(ddof=0))
    loaded, units = data.read_channels(file, file.start_date, file.end_date, [(data.channels[0], "A")])["physical"]
    pd.testing.assert_frame_equal(loaded, raw)


def test_final_channel_callback_rejection_is_window_only(recording):
    calls = []

    def reject(channels):
        calls.append(list(channels))
        return None

    data = dataset([ChannelConfig("signal", ["A"], preprocessors=[RecordingZScore()])], prepare_channels=reject, input_channels=["signal"])
    assert data.get_input_channels() == ["signal"]
    assert calls == []
    file = data.prepare_patient(recording, raise_errors=True)
    assert calls == []
    assert data.get_item(file, file.start_date) is None
    assert calls == [["signal"]]
    assert data.prepare_patient(recording) is not None


def test_final_channel_callback_requires_declared_inputs():
    callback = Mock(side_effect=AssertionError("Construction must not call the mapper"))
    with pytest.raises(ValueError, match="explicit input_channels"):
        dataset([ChannelConfig("source", ["A"])], prepare_channels=callback)
    callback.assert_not_called()


def test_missing_declared_mapper_output_is_an_error(recording):
    data = dataset([ChannelConfig("source", ["A"])], prepare_channels=lambda channels: {"other": channels["source"]}, input_channels=["expected"])
    assert data.get_input_channels() == ["expected"]
    file = data.prepare_patient(recording, raise_errors=True)
    with pytest.raises(KeyError, match="expected"):
        data.get_item(file, file.start_date)


def test_final_channel_callback_combines_physical_and_digital_sources(recording):
    calls = []

    def check_units(channels):
        calls.append({name: unit for name, (values, unit) in channels.items()})
        return channels

    data = dataset([ChannelConfig("physical", ["A"], preprocessors=[RecordingZScore()]), ChannelConfig("digital", ["A"], read_mode="digital", preprocessors=[RecordingZScore()])], prepare_channels=check_units, input_channels=["physical", "digital"])
    loaded = cloudpickle.loads(cloudpickle.dumps(data))
    file = loaded.prepare_patient(recording, raise_errors=True)
    item = loaded.get_item(file, file.start_date)
    np.testing.assert_allclose(item["data"][:, 0], item["data"][:, 1], atol=1e-6)
    file = data.prepare_patient(recording, raise_errors=True)
    assert calls == []
    data.get_item(file, file.start_date)
    assert calls == [{"physical": "dimensionless", "digital": "dimensionless"}]


def test_final_channel_callback_receives_counts_but_not_quality_signals(recording):
    calls = []

    def prepare(channels):
        assert set(channels) == {"signal"}
        assert channels["signal"][1] == "counts"
        calls.append(len(channels["signal"][0]))
        return channels

    data = dataset([ChannelConfig("signal", ["A"], quality_name="B", read_mode="digital")], prepare_channels=prepare, input_channels=["signal"])
    file = data.prepare_patient(recording, raise_errors=True)
    assert data.get_item(file, file.start_date) is not None
    assert calls == [199]


@pytest.mark.parametrize(("length", "expected"), [(3, [-1., 0., 1., 1., 1.]), (7, [-3., -2., -1., 0., 1.])])
def test_window_processors_see_actual_samples_before_final_padding_or_truncation(length, expected):
    calls = []

    def center(values, *, unit, is_recording):
        calls.append(len(values))
        return values - values.mean(), unit

    data = UnlabelledDataset(channels=[ChannelConfig("signal", ["A"], preprocessors=[center])], sample_frequency=1, total_input="5s")
    start = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame({"A": np.arange(length, dtype=float)}, index=pd.date_range(start, periods=length, freq="1s"))
    file = EDFFile(channels=["A"], path="recording.edf", start_date=start, preprocessors={("signal", "A"): [center]})
    file.get_x = lambda *args, **kwargs: frame.copy()
    raw, _ = data.read_channels(file, start, start + pd.Timedelta("5s"), [(data.channels[0], "A")])["physical"]
    pd.testing.assert_frame_equal(raw, frame)
    actual = data.get_item(file, start)["data"][:, 0]
    np.testing.assert_allclose(actual, expected)
    assert calls == [length]


@pytest.mark.parametrize("strategy", ["first", "random"])
def test_missing_configured_quality_channel_still_raises(recording, strategy):
    data = dataset([ChannelConfig("signal", ["A"], quality_name="missing")], group_sampling_strategy=strategy)
    file = data.prepare_patient(recording, raise_errors=True)
    with pytest.raises(ValueError, match="Missing quality channel 'missing'"):
        data.get_item(file, file.start_date)


class ThreeChannelClassifier(BaseModel, ClassifierModel):
    def compute(self, values):
        mean = values.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        return torch.stack((mean, -mean), dim=-1)

    def input_spec(self):
        return (1, 200, 3), {"layout": "BTC", "input_channels": ["difference", "average", "selected"]}


def test_ten_sources_become_three_model_channels_in_metadata_and_package(tmp_path):
    path = write_edf(tmp_path / "ten.edf", [{"label": f"A{i}", "samples": i * 100} for i in range(10)])
    calls = []

    def reduce_channels(channels):
        assert list(channels) == [f"source{i}" for i in range(10)]
        assert all(unit == "uV" for values, unit in channels.values())
        calls.append(len(channels))
        return {
            "unused": channels["source0"],
            "selected": channels["source9"],
            "average": ((channels["source2"][0] + channels["source3"][0]) / 2, "uV"),
            "difference": (channels["source0"][0] - channels["source1"][0], "uV"),
        }

    data = dataset([ChannelConfig(f"source{i}", [f"A{i}"], preprocessors=[ConvertUnit("uV")]) for i in range(10)], prepare_channels=reduce_channels, input_channels=["difference", "average", "selected"])
    assert data.get_input_channels() == ["difference", "average", "selected"]
    package_template = UnlabelledDataset.from_dataset(data)
    assert package_template.get_input_channels() == data.get_input_channels()
    assert calls == []
    data.initialize([path], num_workers=0, strict=True)
    assert calls == []  # Initialization still does not read stateless signals.
    assert data.get_input_channels() == ["difference", "average", "selected"]
    assert calls == []
    item = data.get_item(data.edf_files[0], data.edf_files[0].start_date)
    assert calls == [10]
    assert item["data"].shape == (200, 3)
    assert item["data"][0, 0] < 0 < item["data"][0, 1] < item["data"][0, 2]
    source_count = len(data.channels)
    assert source_count == 10
    contract = {"type": "single-head-multiclass", "classes": ["positive", "negative"], "sequence_len": 1, "target_resolution": "5s"}
    package = PackagedModel(name="three", model=ThreeChannelClassifier(), dataset=UnlabelledDataset.from_dataset(data), task="test", classification_contract=contract)
    candidate = data.clone(channels=data.channels)
    package.assert_compatible(candidate)  # Names are declared; compatibility does not read data.
    candidate.initialize([path], num_workers=0, strict=True)
    package.assert_compatible(candidate)
    assert candidate.get_input_channels() == ["difference", "average", "selected"]
    wrong = UnlabelledDataset(**{**data.dataset_kwargs(), "input_channels": ["wrong"]})
    with pytest.raises(ValueError, match="Expected logical input channels"):
        package.assert_compatible(wrong)
    before = package.predict_patient(path, progress=False)
    loaded = PackagedModel.load(package.save(tmp_path / "three-package"))
    assert loaded.dataset.get_input_channels() == ["difference", "average", "selected"]
    assert len(loaded.dataset.channels) == source_count
    after = loaded.predict_patient(path, progress=False)
    pd.testing.assert_frame_equal(before, after)


def test_final_callback_runs_after_processors_and_before_padding_or_truncation():
    def square(channels):
        values, unit = channels["source"]
        assert values.shape == (3, 1)
        np.testing.assert_allclose(values[:, 0], [-1., 0., 1.])
        return {"output": (values ** 2, unit)}

    data = UnlabelledDataset(channels=[ChannelConfig("source", ["A"], preprocessors=[WindowCenter()])], sample_frequency=1, total_input="5s", prepare_channels=square, input_channels=["output"])
    start = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame({"A": [1., 2., 3.]}, index=pd.date_range(start, periods=3, freq="1s"))
    file = EDFFile(channels=["A"], path="recording.edf", start_date=start, preprocessors={("source", "A"): [WindowCenter()]})
    file.get_x = lambda *args, **kwargs: frame.copy()
    np.testing.assert_allclose(data.get_item(file, start)["data"][:, 0], [1., 0., 1., 1., 1.])


def test_final_callback_can_shorten_values_before_padding():
    data = UnlabelledDataset(channels=[ChannelConfig("source", ["A"])], sample_frequency=1, total_input="5s", prepare_channels=lambda channels: {"output": (channels["source"][0][:3], "uV")}, input_channels=["output"])
    start = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame({"A": [1., 2., 3., 4., 5.]}, index=pd.date_range(start, periods=5, freq="1s"))
    file = EDFFile(channels=["A"], path="recording.edf", start_date=start, preprocessors={("source", "A"): []})
    file.get_x = lambda *args, **kwargs: frame.copy()
    np.testing.assert_allclose(data.get_item(file, start)["data"][:, 0], [1., 2., 3., 3., 3.])


def test_paired_callbacks_choose_different_outputs_from_shared_source_preparation(recording):
    source = dataset([ChannelConfig("source", ["A"])], prepare_channels=lambda channels: {"single": channels["source"]}, input_channels=["single"])
    target = dataset([ChannelConfig("source", ["A"])], prepare_channels=lambda channels: {"first": channels["source"], "second": channels["source"]}, input_channels=["first", "second"])
    paired = PairedDataset({"single": source, "double": target}, base=dataset([]), input_offsets={"single": [0], "double": [0]})
    paired.initialize([recording], num_workers=0, strict=True)
    assert paired.component_files["single"][str(recording)] is paired.component_files["double"][str(recording)]
    assert target.initialized is False  # Source EDF preparation was reused.
    assert paired.get_input_channels() == {"single": ["single"], "double": ["first", "second"]}
    assert paired.input_spec() == {"single": (1, 1, 200, 1), "double": (1, 1, 200, 2)}
    item = paired[0]
    assert item["data"]["single"].shape == (1, 200, 1)
    assert item["data"]["double"].shape == (1, 200, 2)
