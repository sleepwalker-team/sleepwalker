from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_split_tool():
    spec = importlib.util.spec_from_file_location("sleepwalker_split_tool", REPO_ROOT / "tools" / "split.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_holdout_keeps_subject_sessions_together():
    split_tool = load_split_tool()
    files = [
        "/repo/sub-01/ses-1/a.edf",
        "/repo/sub-01/ses-2/b.edf",
        "/repo/sub-02/ses-1/c.edf",
        "/repo/sub-03/ses-1/d.edf",
        "/repo/sub-04/ses-1/e.edf",
    ]

    manifest = split_tool.build_holdout(files, (0.5, 0.25, 0.25), "seed")
    split = manifest["folds"]["holdout"]

    assert set(manifest) == {"seed", "folds"}
    assert sorted(path for paths in split.values() for path in paths) == sorted(files)
    subject_roles = [role for role, paths in split.items() if any("sub-01" in path for path in paths)]
    assert len(subject_roles) == 1


def test_cross_validation_uses_every_subject_for_test_once():
    split_tool = load_split_tool()
    files = [f"/repo/sub-{index:02d}/record.edf" for index in range(9)]

    manifest = split_tool.build_cross_validation(files, 3, "seed")

    assert list(manifest["folds"]) == ["fold_0", "fold_1", "fold_2"]
    assert sorted(path for split in manifest["folds"].values() for path in split["test"]) == sorted(files)
    for split in manifest["folds"].values():
        assert set(split["train"]).isdisjoint(split["validation"])
        assert set(split["train"]).isdisjoint(split["test"])
        assert set(split["validation"]).isdisjoint(split["test"])


def test_cross_validation_requires_three_folds():
    split_tool = load_split_tool()

    with pytest.raises(ValueError, match="at least three"):
        split_tool.build_cross_validation(["a.edf", "b.edf", "c.edf"], 2, "seed")


def test_main_writes_manifest(monkeypatch, tmp_path):
    split_tool = load_split_tool()
    output = tmp_path / "nested" / "split.yml"
    monkeypatch.setattr(split_tool, "find_edf_files", lambda root: [f"/repo/sub-{index}/record.edf" for index in range(10)])
    monkeypatch.setattr(sys, "argv", ["split.py", "/repo", str(output), "--fractions", "0.8", "0.1", "0.1"])

    split_tool.main()

    manifest = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert set(manifest) == {"seed", "folds"}
    assert list(manifest["folds"]) == ["holdout"]


def test_edf_filter_requires_every_channel_group(monkeypatch):
    split_tool = load_split_tool()
    work = []

    def filter_result(item):
        work.append(item)
        path, _, _, _ = item
        return path, "usable" if path.endswith("usable.edf") else "missing_required_channel"

    monkeypatch.setattr(split_tool, "edf_filter_result", filter_result)
    selected = split_tool.filter_edf_files(
        ["usable.edf", "missing.edf"],
        channels={"EEG": ["C3-M2", "C4-M1"], "SpO2": ["SaO2", "SpO2"]},
        min_duration="30min",
        num_workers=1,
    )

    assert selected == ["usable.edf"]
    assert work[0][1] == (("C3-M2", "C4-M1"), ("SaO2", "SpO2"))
    assert work[0][2] == 1800


def test_main_filters_before_assigning_subjects(monkeypatch, tmp_path):
    split_tool = load_split_tool()
    output = tmp_path / "split.yml"
    filter_config = tmp_path / "filters.yml"
    filter_config.write_text("patient_filter: example.filter\n", encoding="utf-8")
    candidates = [f"/repo/sub-{index}/record.edf" for index in range(10)]
    selected = candidates[:5]
    calls = []
    monkeypatch.setattr(split_tool, "find_edf_files", lambda root: candidates)
    monkeypatch.setattr(split_tool, "apply_patient_filter", lambda spec, files, dataset, num_workers: calls.append((spec, files, dataset, num_workers)) or selected)
    monkeypatch.setattr(sys, "argv", ["split.py", "/repo", str(output), "--filter-config", str(filter_config), "--workers", "3"])

    split_tool.main()

    manifest = yaml.safe_load(output.read_text(encoding="utf-8"))
    assigned = [path for partition in manifest["folds"]["holdout"].values() for path in partition]
    assert sorted(assigned) == sorted(selected)
    assert calls == [("example.filter", candidates, None, 3)]
