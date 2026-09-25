"""Process-shared cache for complete resampled EDF recordings."""

from __future__ import annotations

import multiprocessing
import os
from collections import OrderedDict
from pathlib import Path
import shutil
import tempfile
import time
from typing import Callable, Hashable, Sequence
import uuid

import numpy as np
import pandas as pd


class EDFCacheEntry:
    """Expose one process-local mapping of a globally cached recording."""

    def __init__(self, cache, key: Hashable, metadata: dict, array: np.ndarray):
        self.cache = cache
        self.key = key
        self.metadata = metadata
        self.array = array

    def __enter__(self):
        return self

    def copy_window(self, start: pd.Timestamp, end: pd.Timestamp, channels: Sequence[str]) -> pd.DataFrame:
        if self.array is None:
            raise RuntimeError("EDF cache entry must be entered before it can be read.")
        available = list(self.metadata["columns"])
        missing = sorted(set(channels) - set(available))
        if missing:
            raise ValueError(f"Cached EDF recording is missing channels {missing}.")
        period_ns = int(self.metadata["period_ns"])
        origin_ns = int(self.metadata["start_ns"])
        start_ns = pd.Timestamp(start).value
        end_ns = pd.Timestamp(end).value
        first = max((start_ns - origin_ns + period_ns - 1) // period_ns, 0)
        stop = min(max((end_ns - origin_ns + period_ns - 1) // period_ns, first + 1), self.array.shape[0])
        column_indices = [available.index(channel) for channel in channels]
        values = self.cache.copy_window(self.array, slice(first, stop), column_indices)
        index_start = pd.Timestamp(origin_ns + first * period_ns)
        index = pd.date_range(start=index_start, periods=len(values), freq=pd.to_timedelta(period_ns, unit="ns"))
        return pd.DataFrame(values, columns=list(channels), index=index)

    def __exit__(self, exception_type, exception, traceback):
        pass


class EDFCache:
    """Share a patient-bounded LRU cache between DataLoader worker processes.

    Complete recordings are stored as NumPy memory maps. A manager-backed
    registry coordinates cache misses and leases; signal samples never pass
    through the manager process.
    """

    def __init__(self, max_patients: int, directory: str | os.PathLike | None = None, loading_timeout: float = 3600.0):
        if max_patients < 1:
            raise ValueError("max_patients must be positive.")
        if loading_timeout <= 0:
            raise ValueError("loading_timeout must be positive.")
        root = Path(directory) if directory is not None else Path("/dev/shm")
        if not root.is_dir() or not os.access(root, os.W_OK):
            root = Path(tempfile.gettempdir())
        self.directory = Path(tempfile.mkdtemp(prefix="sleepwalker-edf-cache-", dir=root))
        self.max_patients = int(max_patients)
        self.loading_timeout = float(loading_timeout)
        self.owner_pid = os.getpid()
        self.manager = multiprocessing.get_context("spawn").Manager()
        self.entries = self.manager.dict()
        self.lock = self.manager.RLock()
        self.clock = self.manager.Value("q", 0)
        self.local_entries = {}
        self.local_patients = OrderedDict()
        self.closed = False

    def __getstate__(self):
        state = dict(self.__dict__)
        state["manager"] = None
        state["local_entries"] = {}
        state["local_patients"] = OrderedDict()
        return state

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()

    def next_access(self) -> int:
        self.clock.value += 1
        return int(self.clock.value)

    def store_frame(self, frame: pd.DataFrame) -> dict:
        if frame.empty:
            raise ValueError("Cannot cache an empty EDF recording.")
        if len(frame.index) < 2:
            raise ValueError("A cached EDF recording requires at least two samples.")
        periods = np.diff(frame.index.asi8)
        if not np.all(periods == periods[0]):
            raise ValueError("Cached EDF recordings must have a uniform sample period.")
        path = self.directory / f"{uuid.uuid4().hex}.npy"
        values = np.ascontiguousarray(frame.to_numpy(), dtype=np.float32)
        stored = np.lib.format.open_memmap(path, mode="w+", dtype=values.dtype, shape=values.shape)
        stored[:] = values
        stored.flush()
        mmap = getattr(stored, "_mmap", None)
        if mmap is not None:
            mmap.close()
        return {
            "file": str(path),
            "columns": tuple(frame.columns),
            "start_ns": int(frame.index[0].value),
            "period_ns": int(periods[0]),
            "shape": tuple(values.shape),
            "dtype": values.dtype.str,
        }

    def acquire(self, key: Hashable, patient: str, loader: Callable[[], pd.DataFrame]) -> EDFCacheEntry:
        """Acquire a recording, populating one shared miss in the calling worker."""
        local = self.local_entries.get(key)
        if local is not None:
            self.touch_local_patient(patient)
            return EDFCacheEntry(self, key, local["metadata"], local["array"])

        loading_started = time.monotonic()
        while True:
            populate = False
            with self.lock:
                metadata = self.entries.get(key)
                if metadata is None:
                    self.entries[key] = {"state": "loading", "patient": patient, "owner": os.getpid(), "started": time.time()}
                    populate = True
                elif metadata["state"] == "ready":
                    metadata = dict(metadata)
                    metadata["access"] = self.next_access()
                    self.entries[key] = metadata
                    return self.attach(key, metadata)
                elif metadata["state"] != "loading":
                    raise RuntimeError(f"Unknown EDF cache entry state {metadata['state']!r}.")

            if populate:
                try:
                    stored = self.store_frame(loader())
                    with self.lock:
                        metadata = {**stored, "state": "ready", "patient": patient, "access": self.next_access()}
                        self.entries[key] = metadata
                        self.evict()
                        return self.attach(key, metadata)
                except Exception:
                    with self.lock:
                        current = self.entries.get(key)
                        if current is not None and current["state"] == "loading" and current["owner"] == os.getpid():
                            del self.entries[key]
                    raise

            if time.monotonic() - loading_started > self.loading_timeout:
                raise TimeoutError(f"Timed out waiting for EDF cache entry populated by process {metadata['owner']}.")
            time.sleep(0.05)

    def copy_window(self, array: np.ndarray, rows: slice, columns: Sequence[int]) -> np.ndarray:
        """Copy one requested window out of shared storage before releasing it."""
        if list(columns) == list(range(array.shape[1])):
            return np.array(array[rows], copy=True)
        return np.take(array[rows], columns, axis=1)

    def attach(self, key: Hashable, metadata: dict) -> EDFCacheEntry:
        """Open one shared recording once in this process and retain its mapping."""
        patient = metadata["patient"]
        if patient not in self.local_patients and len(self.local_patients) >= self.max_patients:
            oldest_patient, _ = self.local_patients.popitem(last=False)
            self.close_local_patient(oldest_patient)
        array = np.load(metadata["file"], mmap_mode="r")
        self.local_entries[key] = {"metadata": metadata, "array": array}
        self.touch_local_patient(patient)
        return EDFCacheEntry(self, key, metadata, array)

    def touch_local_patient(self, patient: str) -> None:
        """Mark one process-local patient attachment as recently used."""
        self.local_patients.pop(patient, None)
        self.local_patients[patient] = None

    def close_local_patient(self, patient: str) -> None:
        """Close every process-local mapping belonging to one patient."""
        keys = [key for key, local in self.local_entries.items() if local["metadata"]["patient"] == patient]
        for key in keys:
            local = self.local_entries.pop(key)
            mmap = getattr(local["array"], "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.local_patients.pop(patient, None)

    def close_local(self) -> None:
        """Close all memory maps owned by this process."""
        for patient in list(self.local_patients):
            self.close_local_patient(patient)

    def evict(self) -> None:
        """Evict least-recently-used patients whose entries have no leases."""
        ready = [(key, dict(metadata)) for key, metadata in self.entries.items() if metadata["state"] == "ready"]
        patients = {metadata["patient"] for _, metadata in ready}
        while len(patients) > self.max_patients:
            candidates = []
            for patient in patients:
                entries = [(key, metadata) for key, metadata in ready if metadata["patient"] == patient]
                if entries:
                    candidates.append((max(metadata["access"] for _, metadata in entries), patient, entries))
            if not candidates:
                return
            _, patient, patient_entries = min(candidates, key=lambda candidate: candidate[0])
            for key, metadata in patient_entries:
                self.entries.pop(key, None)
                Path(metadata["file"]).unlink(missing_ok=True)
                ready.remove((key, metadata))
            patients.remove(patient)

    def close(self) -> None:
        """Remove all cache files and stop the shared registry manager."""
        if self.closed:
            return
        self.close_local()
        if os.getpid() != self.owner_pid:
            self.closed = True
            return
        self.closed = True
        with self.lock:
            self.entries.clear()
        shutil.rmtree(self.directory, ignore_errors=True)
        self.manager.shutdown()
