import numpy as np
import pandas as pd
import multiprocessing

from sleepwalker.datasets.EDFCache import EDFCache


def read_cached_window(cache, key, start, queue):
    def fail_load():
        raise AssertionError("Child process unexpectedly missed the shared cache entry.")

    with cache.acquire(key, "patient", fail_load) as entry:
        window = entry.copy_window(start, start + pd.Timedelta("1s"), ["EEG"])
    queue.put(window["EEG"].tolist())


def test_edf_cache_populates_once_and_returns_owned_windows(tmp_path):
    start = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame({"EEG": np.arange(100, dtype=np.float32)}, index=pd.date_range(start, periods=100, freq="100ms"))
    loads = []

    def load():
        loads.append(True)
        return frame

    with EDFCache(max_patients=1, directory=tmp_path) as cache:
        with cache.acquire(("patient", 10), "patient", load) as entry:
            first = entry.copy_window(start + pd.Timedelta("1s"), start + pd.Timedelta("2s"), ["EEG"])
        with cache.acquire(("patient", 10), "patient", load) as entry:
            second = entry.copy_window(start + pd.Timedelta("1s"), start + pd.Timedelta("2s"), ["EEG"])

        assert len(loads) == 1
        assert first.equals(second)
        assert first["EEG"].tolist() == list(np.arange(10, 20, dtype=np.float32))
        first.iloc[0, 0] = -1
        assert second.iloc[0, 0] == 10


def test_edf_cache_is_shared_with_spawned_workers(tmp_path):
    start = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame({"EEG": np.arange(100, dtype=np.float32)}, index=pd.date_range(start, periods=100, freq="100ms"))
    key = ("patient", 10)

    with EDFCache(max_patients=1, directory=tmp_path) as cache:
        with cache.acquire(key, "patient", lambda: frame):
            pass
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        process = context.Process(target=read_cached_window, args=(cache, key, start, queue))
        process.start()
        process.join(timeout=30)

        assert process.exitcode == 0
        assert queue.get(timeout=1) == list(np.arange(10, dtype=np.float32))
