from pathlib import Path

import numpy as np
import pytest

from sleepwalker.server.services.signal_reader import EdfSignalReader, SignalSamples, downsample_extrema


def test_downsample_extrema_preserves_boundaries_and_strongest_values() -> None:
    samples = SignalSamples(
        times=[float(index) for index in range(10)],
        values=[0.0, 1.0, -8.0, 2.0, 1.0, 9.0, 2.0, -7.0, 1.0, 0.0],
    )

    reduced = downsample_extrema(samples, max_points=4)

    assert reduced.times[0] == 0.0
    assert reduced.times[-1] == 9.0
    assert len(reduced.times) == 4
    assert -8.0 in reduced.values
    assert 9.0 in reduced.values


class ArrayReader:
    def __init__(self, values, sample_rate=10.0):
        self.values = np.asarray(values, dtype=float)
        self.sample_rate = sample_rate
        self.closed = False
        self.read_sizes = []

    def getSignalLabels(self):
        return ["EEG"]

    def getSampleFrequency(self, _index):
        return self.sample_rate

    def getNSamples(self):
        return [len(self.values)]

    def readSignal(self, _index, start, n):
        self.read_sizes.append(n)
        assert n <= 7
        return self.values[start:start + n]

    def close(self):
        self.closed = True


@pytest.mark.parametrize("max_points", [2, 3, 4, 7, 23, 200])
def test_streaming_reader_matches_existing_extrema_selection(max_points, monkeypatch):
    values = np.sin(np.arange(120))
    values[19] = 99.0
    values[20] = -99.0
    reader = ArrayReader(values)
    monkeypatch.setattr("sleepwalker.server.services.signal_reader.pyedflib.EdfReader", lambda _: reader)
    monkeypatch.setattr("sleepwalker.server.services.signal_reader.SIGNAL_CHUNK_SIZE", 7)
    start, end = 0.15, 11.95
    indices = [index for index in range(len(values)) if start <= index / 10.0 < end]
    baseline = SignalSamples(
        times=[index / 10.0 for index in indices],
        values=[float(values[index]) for index in indices],
    )
    expected = downsample_extrema(baseline, max_points)
    actual = EdfSignalReader().read(Path("test.edf"), "EEG", start, end, max_points=max_points)
    assert actual.times == expected.times
    assert actual.values == expected.values
    assert actual.sample_count == len(indices)
    assert len(actual.times) <= max_points
    assert reader.closed


def test_large_signal_is_read_in_bounded_chunks_not_as_one_array(monkeypatch):
    class LargeReader(ArrayReader):
        def __init__(self):
            super().__init__([])

        def getNSamples(self):
            return [10_000_000]

        def readSignal(self, _index, start, n):
            assert n <= 65536
            self.read_sizes.append(n)
            return np.arange(start, start + n, dtype=float)

    reader = LargeReader()
    monkeypatch.setattr("sleepwalker.server.services.signal_reader.pyedflib.EdfReader", lambda _: reader)
    samples = EdfSignalReader().read(Path("large.edf"), "EEG", 0.0, 1_000_000.0, max_points=128)
    assert len(samples.times) == len(samples.values) == 128
    assert samples.sample_count == 10_000_000
    assert sum(reader.read_sizes) == 10_000_000
    assert samples.times[0] == 0.0
    assert samples.times[-1] == 999_999.9
    assert reader.closed


def test_reader_closes_file_on_midstream_failure(monkeypatch):
    reader = ArrayReader(np.arange(100))

    def fail(_index, start, n):
        raise RuntimeError("read failed")

    reader.readSignal = fail
    monkeypatch.setattr("sleepwalker.server.services.signal_reader.pyedflib.EdfReader", lambda _: reader)
    with pytest.raises(RuntimeError, match="read failed"):
        EdfSignalReader().read(Path("test.edf"), "EEG", 0.0, 10.0, max_points=8)
    assert reader.closed
