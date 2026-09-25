from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_estimator():
    path = REPO_ROOT / "iclr2026" / "scripts" / "estimate_signal_distribution.py"
    spec = importlib.util.spec_from_file_location("iclr_signal_distribution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_estimator_contains_one_static_group_per_logical_signal_source():
    estimator = load_estimator()

    assert set(estimator.SIGNAL_GROUPS) == {"eeg", "eog", "chin_emg", "ecg", "abdomen", "chest", "airflow", "spo2"}
    assert "C3-M2" in estimator.SIGNAL_GROUPS["eeg"]["physical_names"]
    assert "E1-M2" in estimator.SIGNAL_GROUPS["eog"]["physical_names"]
    assert estimator.SIGNAL_GROUPS["abdomen"]["physical_names"] == ["ABD", "ABDOMEN", "Abdomen"]
    assert estimator.SIGNAL_GROUPS["chest"]["physical_names"] == ["CHEST", "THORAX", "Chest"]
    assert "PTAF" in estimator.SIGNAL_GROUPS["airflow"]["physical_names"]
    assert estimator.SIGNAL_GROUPS["spo2"]["unit"] == "%"
    assert all(group["normalizer"].normalize is False for group in estimator.SIGNAL_GROUPS.values())


def test_estimator_merges_patient_moments_and_reports_mean_std():
    estimator = load_estimator()
    total = {"eeg": estimator.empty_statistics()}
    profiles = {"eeg": {"unit": "uV", "normalizer": estimator.EEGFilterNormalizer(fs=100, normalize=False)}}
    first = {
        "eeg": {
            "count": 2,
            "sum": 4.0,
            "sum_squares": 10.0,
            "patient_count": 1,
            "channel_count": 1,
            "physical_channels": {"C3-M2": 1},
        }
    }
    second = {
        "eeg": {
            "count": 2,
            "sum": 8.0,
            "sum_squares": 34.0,
            "patient_count": 1,
            "channel_count": 1,
            "physical_channels": {"C4-M1": 1},
        }
    }

    estimator.merge_statistics(total, first)
    estimator.merge_statistics(total, second)
    result = estimator.finalize_statistics(total, profiles)["eeg"]

    assert result["mean"] == 3.0
    assert np.isclose(result["std"], np.sqrt(2.0))
    assert result["n_samples"] == 4
    assert result["n_patients"] == 2
    assert result["physical_channels"] == {"C3-M2": 1, "C4-M1": 1}


def test_estimate_patient_loads_all_required_channels_once(monkeypatch):
    estimator = load_estimator()
    profiles = {
        "first": {
            "physical_names": ["A"],
            "unit": "uV",
            "normalizer": estimator.SignalFilterNormalizer(fs=10, normalize=False),
        },
        "second": {
            "physical_names": ["B"],
            "unit": "uV",
            "normalizer": estimator.SignalFilterNormalizer(fs=10, normalize=False),
        },
    }
    calls = []
    monkeypatch.setattr(
        estimator,
        "read_edf_meta",
        lambda path: {"source": "pyedflib", "signals": ["A", "B"], "units": {"A": "uV", "B": "uV"}},
    )
    monkeypatch.setattr(
        estimator,
        "edf_to_df",
        lambda path, channels, **kwargs: calls.append(list(channels)) or pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 5.0]}),
    )

    result = estimator.estimate_patient(
        "patient.edf",
        signal_groups=profiles,
    )

    assert calls == [["A", "B"]]
    assert result["first"]["sum"] == 3.0
    assert result["second"]["sum_squares"] == 34.0
