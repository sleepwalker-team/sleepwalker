from pathlib import Path

import numpy as np
from pyedflib import highlevel
import pytest

from sleepwalker.server.recording.relabel import channel_renames, read_edf_labels, relabel_edf


@pytest.fixture
def edf(tmp_path) -> Path:
    path = tmp_path / "source.edf"
    labels = ["EEG Fpz-Cz", "C4-M1", "EOG horizontal"]
    signals = [np.sin(np.arange(1000) / (10 + index)) for index in range(len(labels))]
    headers = highlevel.make_signal_headers(labels, dimension="uV", sample_frequency=100, physical_min=-2, physical_max=2)
    highlevel.write_edf(str(path), signals, headers)
    return path


def test_relabel_changes_only_the_header_labels(edf, tmp_path):
    target = tmp_path / "relabeled.edf"
    relabel_edf(edf, target, {"EEG Fpz-Cz": "C3-M2"})
    # pyedflib writes EDF+, which adds an annotation signal.
    assert read_edf_labels(target) == ["C3-M2", "C4-M1", "EOG horizontal", "EDF Annotations"]
    source, copy = edf.read_bytes(), target.read_bytes()
    assert len(source) == len(copy)
    labels = slice(256, 256 + 4 * 16)
    assert source[:labels.start] == copy[:labels.start] and source[labels.stop:] == copy[labels.stop:]
    signals, headers, _ = highlevel.read_edf(str(target))
    assert [header["label"] for header in headers] == ["C3-M2", "C4-M1", "EOG horizontal"]
    assert np.array_equal(signals, highlevel.read_edf(str(edf))[0])


@pytest.mark.parametrize("renames", [
    {"EEG Fpz-Cz": "a label longer than 16"},
    {"EEG Fpz-Cz": "C3–M2"},
    {"EEG Fpz-Cz": "C4-M1"},
    {"unknown": "C3-M2"},
])
def test_relabel_rejects_labels_it_cannot_write(edf, tmp_path, renames):
    with pytest.raises(ValueError):
        relabel_edf(edf, tmp_path / "relabeled.edf", renames)


def test_mapped_signal_gets_first_package_name_and_competitors_are_renamed():
    labels = ["EEG Fpz-Cz", "C4-M1", "EOG horizontal"]
    aliases = {"eeg": ("C3-M2", "C4-M1")}
    renames = channel_renames(labels, aliases, {"eeg": "EEG Fpz-Cz"})
    assert renames == {"EEG Fpz-Cz": "C3-M2", "C4-M1": "unused 0"}
