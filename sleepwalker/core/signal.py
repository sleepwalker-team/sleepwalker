from __future__ import annotations

from abc import ABC, abstractmethod
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scipy.signal import butter, filtfilt, iirnotch

import numpy as np
import pandas as pd

import pyedflib
from pyedflib import DO_NOT_READ_ANNOTATIONS, DO_NOT_CHECK_FILE_SIZE

def read_edf_meta(edf_path: str) -> Dict[str, Any]:
    """Read basic metadata about an EDF file.

    Returns dict with keys: start, end, duration_s, labels, sample_rates.
    """
    with pyedflib.EdfReader(edf_path, annotations_mode=DO_NOT_READ_ANNOTATIONS, check_file_size=DO_NOT_CHECK_FILE_SIZE) as f: 
        labels = f.getSignalLabels()
        fs = {lab: float(f.getSampleFrequency(i)) for i, lab in enumerate(labels)}
        duration_s = float(f.getFileDuration())
        start = f.getStartdatetime()
        end = start + pd.to_timedelta(f"{duration_s} s")
        return {
            "start": start,
            "end": end,
            "duration_s": duration_s,
            "signals": labels,
            "fs": fs,
        }

def edf_to_df(
    edf_path: str,
    channels: List[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    frequency: float, 
    how: str = "nearest"
) -> pd.DataFrame:
    """Read raw samples for a single channel between [start, end).

    Returns (signal, original_dt) where original_dt is the sampling period.
    """
    with pyedflib.EdfReader(edf_path, annotations_mode=DO_NOT_READ_ANNOTATIONS, check_file_size=DO_NOT_CHECK_FILE_SIZE) as f:
        labels = f.getSignalLabels()
        
        dfs = []
        for ch in channels:
            if ch in labels:
                idx = labels.index(ch)
                fs = float(f.getSampleFrequency(idx))
                dt = pd.to_timedelta(f"{1.0 / fs}s")
                file_start = pd.Timestamp(f.getStartdatetime())
                duration_s = float(f.getFileDuration())
                file_end = file_start + pd.to_timedelta(f"{duration_s}s")
                if start is None:
                    start = file_start 
                if end is None:
                    end = file_end
                i0 = max(int((start - file_start) / dt), 0)
                i1 = max(int((end - file_start) / dt), i0 + 1)
                x = f.readSignal(idx, start=i0, n=i1 - i0, digital=False)

                df = pd.DataFrame(x, columns=[ch])
                df.index = pd.date_range(start=start, periods=len(x), freq=dt)
                if how == "mean":
                    df = df.resample(pd.to_timedelta(1.0/frequency, unit="s")).mean()
                elif how == "max":
                    df = df.resample(pd.to_timedelta(1.0/frequency, unit="s")).max()
                else: 
                    df = df.resample(pd.to_timedelta(1.0/frequency, unit="s")).nearest()
                dfs.append(df)
        if len(dfs) > 0:
            return pd.concat(dfs, axis=1, join="outer").ffill().bfill()
        else:
            return pd.DataFrame()