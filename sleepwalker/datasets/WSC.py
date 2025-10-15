from __future__ import annotations
from datetime import datetime, timedelta

import os
from pathlib import Path
import re
import pandas as pd
from .Basedataset import BaseDataset  

class WSC(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        folder = Path(edf_path).parent
        name = Path(edf_path).name.split(".edf")[0]
        annot_path = os.path.join(folder, f"{name}.allscore.txt")
        
        # Read file as two columns: time + event description
        df = pd.read_csv(annot_path, sep="\t", engine="python", names=["time", "event"], comment="#")
        df["time"] = df["time"].str.strip()
        df["event"] = df["event"].astype(str).str.strip()

        # Prepare base date (00:00 same day as EDF start)
        base_date = pd.Timestamp(start_datetime).normalize()
        start_times, end_times, labels = [], [], []

        prev_ts = None
        for t_str, ev in zip(df["time"], df["event"]):
            # Parse clock time (e.g., "21:33:01.00")
            try:
                t_obj = pd.to_datetime(t_str, format="%H:%M:%S.%f")
            except ValueError:
                t_obj = pd.to_datetime(t_str, format="%H:%M:%S", errors="coerce")
            ts = pd.Timestamp.combine(base_date.to_pydatetime().date(), t_obj.time())

            # Handle midnight rollover
            if prev_ts is not None and ts < prev_ts:
                ts += pd.Timedelta(days=1)
                base_date += pd.Timedelta(days=1)
            prev_ts = ts

            # --- Case 1: STAGE event ---
            if "stage" in ev.lower():
                label = ev.split("STAGE -")[-1].strip().lower()
                duration = pd.Timedelta(seconds=30)
                start_times.append(ts)
                end_times.append(ts + duration)
                labels.append(label)
                continue

            # --- Case 2: has duration (DUR: X SEC.) ---
            dur_match = re.search(r"DUR:\s*([0-9.]+)\s*SEC", ev, re.IGNORECASE)
            if dur_match:
                duration = pd.to_timedelta(float(dur_match.group(1)), unit="s")
                label = re.split(r"DUR:.*?SEC\.?-", ev, flags=re.IGNORECASE)[-1].strip().lower()
                start_times.append(ts)
                end_times.append(ts + duration)
                labels.append(label)
                continue

            # --- Drop everything else ---
            continue

        df = pd.DataFrame({
            "Label": labels,
            "Starttime": start_times,
            "Endtime": end_times,
            "Duration": [e - s for s, e in zip(start_times, end_times)]
        })

        return df[["Label", "Starttime", "Endtime", "Duration"]]