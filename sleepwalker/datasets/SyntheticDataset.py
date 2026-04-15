"""Small synthetic dataset adapter used in tests and local experiments."""

from __future__ import annotations
from pathlib import Path
import re

import numpy as np
import pandas as pd
from typing import Optional
from .Basedataset import BaseDataset  

def load_stages(edf_path: str, start_datetime: pd.Timestamp, basename:str = "stages") -> pd.DataFrame:
    """
    Load sleep stages corresponding to an EDF file.

    Args:
        edf_path: Path to EDF file, e.g. ".../signals_03.edf"
        start_datetime: Start datetime of the EDF recording

    Returns:
        pd.DataFrame with columns [Label, Starttime, Endtime, Duration]
    """
    edf_path = Path(edf_path)

    # extract patient ID (assumes "signals_0X.edf")
    m = re.search(r"signals_(\d+)\.edf$", edf_path.name)
    if not m:
        raise ValueError(f"Invalid EDF filename format: {edf_path}")
    pid = m.group(1)

    # construct stage file path
    stage_path = edf_path.with_name(f"{basename}_{pid}.txt")
    if not stage_path.exists():
        raise FileNotFoundError(f"Stage file not found: {stage_path}")

    # read stage file: each line "start_offset,end_offset,label"
    rows = []
    with open(stage_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            start_off, end_off, label = line.split(",")
            start_off, end_off, label = int(start_off), int(end_off), label

            start = start_datetime + pd.to_timedelta(start_off, unit="s")
            end = start_datetime + pd.to_timedelta(end_off, unit="s")
            duration = (end - start).total_seconds()

            rows.append({
                "Label": label,
                "Starttime": start,
                "Endtime": end,
                "Duration": duration,
            })

    return pd.DataFrame(rows)

class SyntheticDataset(BaseDataset):
    """
    Synthetic dataset for testing models and trainers without EDF dependencies.

    It simulates EDF-like inputs and event DataFrames. Both main events
    and optional extra events can be generated.
    """

    def __init__(self, has_extra: bool = False, **kwargs):
        """Construct a test-oriented dataset backed by local stage text files.

        Args:
            has_extra: Whether `get_extra_event_df` should expose the
                `extra_stages_*.txt` files.
            **kwargs: Forwarded to `BaseDataset`.
        """
        self._has_extra = has_extra
        super().__init__(**kwargs)

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        """Load the primary stage annotations for one synthetic EDF path."""
        return load_stages(edf_path=edf_path, start_datetime=start_datetime)

    def get_extra_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        """Load the secondary stage annotations when enabled."""
        if not self._has_extra:
            return pd.DataFrame()
        else:
            return load_stages(edf_path=edf_path, start_datetime=start_datetime, basename="extra_stages")

    def has_extra_target(self) -> bool:
        """Return whether this dataset exposes secondary target annotations."""
        return self._has_extra
