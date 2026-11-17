from __future__ import annotations

from datetime import datetime
from functools import partial
from pathlib import Path
import warnings

import numpy as np
import pyedflib
import pytest

from sleepwalker.datasets.normalizer import ConvertUnit
from sleepwalker.cli import mirror_edf_repo


def channel_config() -> list[mirror_edf_repo.ChannelConfig]:
    return [
        mirror_edf_repo.ChannelConfig(logical_name='high', physical_names=['HIGH'], preprocessors=[ConvertUnit('uV')]),
        mirror_edf_repo.ChannelConfig(logical_name='low', physical_names=['LOW'], preprocessors=[ConvertUnit('uV')]),
    ]


def write_test_edf(path: Path) -> dict[str, np.ndarray]:
    signals = {
        "HIGH": np.linspace(-0.2, 0.2, 16, dtype=np.float64),
        "LOW": np.array([-20.0, -5.0, 5.0, 20.0], dtype=np.float64),
        "DROP": np.linspace(-1.0, 1.0, 32, dtype=np.float64),
    }
    headers = [
        pyedflib.highlevel.make_signal_header(
            "HIGH",
            dimension="mV",
            sample_frequency=4,
            physical_min=-1,
            physical_max=1,
        ),
        pyedflib.highlevel.make_signal_header(
            "LOW",
            dimension="uV",
            sample_frequency=1,
            physical_min=-100,
            physical_max=100,
        ),
        pyedflib.highlevel.make_signal_header(
            "DROP",
            dimension="uV",
            sample_frequency=8,
            physical_min=-2,
            physical_max=2,
        ),
    ]
    header = pyedflib.highlevel.make_header(
        startdate=datetime(2025, 1, 2, 3, 4, 5),
    )
    header["annotations"] = [[1.0, 0.5, "event"]]
    pyedflib.highlevel.write_edf(
        str(path),
        list(signals.values()),
        headers,
        header=header,
    )
    return signals


def write_config(path: Path) -> None:
    path.write_text(
        """
data:
  name: sleepwalker.datasets.HSP.HSP
  sample_frequency: 2
  channels:
    - logical_name: high
      physical_names: [HIGH]
      preprocessors:
      - name: sleepwalker.datasets.normalizer.ConvertUnit.ConvertUnit
        target: uV
      - name: this.is.deliberately.not.imported
    - logical_name: low
      physical_names: [LOW]
      preprocessors:
      - name: sleepwalker.datasets.normalizer.ConvertUnit.ConvertUnit
        target: uV
""".strip()
        + "\n",
        encoding="utf-8",
    )


def read_rates(path: Path) -> tuple[list[str], dict[str, float]]:
    with pyedflib.EdfReader(str(path)) as reader:
        labels = list(reader.getSignalLabels())
        rates = {
            label: float(reader.getSampleFrequency(index))
            for index, label in enumerate(labels)
        }
    return labels, rates


def test_mirror_downsamples_high_rates_and_copies_other_files(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    nested = source / "sub-01" / "ses-01"
    nested.mkdir(parents=True)
    original = write_test_edf(nested / "record.edf")
    (nested / "annotations.csv").write_text("event,start\n", encoding="utf-8")
    (source / "empty").mkdir()
    config = tmp_path / "train.yml"
    write_config(config)

    return_code = mirror_edf_repo.main([str(source), str(destination), str(config), "--workers", "2"])

    output = destination / "sub-01" / "ses-01" / "record.edf"
    labels, rates = read_rates(output)
    assert labels == ["HIGH", "LOW"]
    assert rates == {"HIGH": 2.0, "LOW": 1.0}
    assert (destination / "sub-01" / "ses-01" / "annotations.csv").read_text() == "event,start\n"
    assert (destination / "empty").is_dir()
    assert return_code == 0
    with pyedflib.EdfReader(str(output)) as reader:
        assert len(reader.readSignal(0)) == 8
        unchanged_low = reader.readSignal(1)
        onsets, durations, descriptions = reader.readAnnotations()
        assert onsets.tolist() == [1.0]
        assert durations.tolist() == [0.5]
        assert descriptions.tolist() == ["event"]
    np.testing.assert_allclose(unchanged_low, original["LOW"], atol=0.01)


def test_exact_sampling_resamples_lower_rate_channels_too(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    write_test_edf(source / "record.edf")

    summary = mirror_edf_repo.mirror_repository(source, destination, channel_config(), 2, exact_sampling=True, workers=1)

    labels, rates = read_rates(destination / "record.edf")
    assert labels == ["HIGH", "LOW"]
    assert rates == {"HIGH": 2.0, "LOW": 2.0}
    with pyedflib.EdfReader(str(destination / "record.edf")) as reader:
        assert len(reader.readSignal(1)) == 8
    assert summary["processed"] == 1


def test_optional_unit_conversion_updates_values_and_header(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    original = write_test_edf(source / "record.edf")

    summary = mirror_edf_repo.mirror_repository(source, destination, channel_config(), 4, convert_units=True, workers=1)

    with pyedflib.EdfReader(str(destination / "record.edf")) as reader:
        assert reader.getPhysicalDimension(0) == "uV"
        converted = reader.readSignal(0)
    np.testing.assert_allclose(converted, original["HIGH"] * 1000, atol=0.1)
    assert summary["processed"] == 1


def test_mirror_skips_a_bad_edf_and_continues(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    write_test_edf(source / "good.edf")
    (source / "bad.edf").write_bytes(b"not an EDF file")

    summary = mirror_edf_repo.mirror_repository(source, destination, channel_config(), 2, workers=1)

    assert summary["processed"] == 1
    assert summary["failed"] == 1
    assert (destination / "good.edf").is_file()
    assert not (destination / "bad.edf").exists()


def test_process_edf_suppresses_only_physical_boundary_warnings(tmp_path, monkeypatch):
    source = tmp_path / "source.edf"
    target = tmp_path / "target.edf"
    write_test_edf(source)
    read_selected_edf = partial(mirror_edf_repo.read_edf, channels={"HIGH": "uV", "LOW": "uV"}, unit_overrides={}, max_sample_rate=2, exact_sampling=False, resample_method="nearest", convert_units=False, assume_units_if_missing=False)
    write_edf = pyedflib.highlevel.write_edf

    def noisy_write_edf(*args, **kwargs):
        warnings.warn("phys_min is -1.0, but signal_min is -1.0 for channel HIGH", UserWarning)
        warnings.warn("unrelated write warning", UserWarning)
        return write_edf(*args, **kwargs)

    monkeypatch.setattr(pyedflib.highlevel, "write_edf", noisy_write_edf)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mirror_edf_repo.process_edf((str(source), str(target)), read_selected_edf, False)

    assert result[0] == "processed"
    messages = [str(warning.message) for warning in caught]
    assert "phys_min is -1.0, but signal_min is -1.0 for channel HIGH" not in messages
    assert "unrelated write warning" in messages


def test_process_edf_writes_outward_rounded_physical_bounds(tmp_path):
    source = tmp_path / "source.edf"
    target = tmp_path / "target.edf"
    source.write_bytes(b"source metadata")
    signal_min = 7.629508999999997
    signal_header = pyedflib.highlevel.make_signal_header("AirFlow", sample_frequency=1, physical_min=7.629509, physical_max=9)

    def read_positive_edf(path):
        signal = np.array([signal_min, 8.5], dtype=np.float64)
        return [signal], [signal_header], pyedflib.highlevel.make_header(), pyedflib.FILETYPE_EDFPLUS

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mirror_edf_repo.process_edf((str(source), str(target)), read_positive_edf, False)

    assert result[0] == "processed"
    assert not any("Physical minimum" in str(warning.message) for warning in caught)
    with pyedflib.EdfReader(str(target)) as reader:
        physical_min = reader.getPhysicalMinimum(0)
    assert physical_min <= signal_min
    assert len(str(signal_header["physical_min"])) <= 8


def test_edf_physical_bounds_round_outward():
    lower = mirror_edf_repo.edf_physical_bound(-2.4143499999999998, lower=True)
    upper = mirror_edf_repo.edf_physical_bound(2.4744070000000002, lower=False)

    assert lower <= -2.4143499999999998
    assert upper >= 2.4744070000000002
    assert len(str(lower)) <= 8
    assert len(str(upper)) <= 8


def test_mirror_rejects_invalid_worker_count(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="workers must be positive"):
        mirror_edf_repo.mirror_repository(source, tmp_path / "destination", channel_config(), 2, workers=0)
