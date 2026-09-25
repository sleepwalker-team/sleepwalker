from __future__ import annotations
from datetime import datetime, timedelta

import pandas as pd
from .Basedataset import BaseDataset  

class Stages(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        """
        Load CSV annotations for the Stages dataset.

        Args:
            edf_path: Path to the EDF file (used to locate .csv annotation file)
            start_datetime: Timestamp indicating when the EDF recording started

        Returns:
            pd.DataFrame with columns:
                ['Label', 'Starttime', 'Endtime', 'Duration']
        """
        annot_path = f"{edf_path.split('.edf')[0]}.csv"
        df = pd.read_csv(annot_path, sep=",")
        df.columns = [c.strip().lower() for c in df.columns]

        # Expected columns: "start time", "duration (seconds)", "event"
        # Parse start times relative to start_date
        current_date = pd.Timestamp(start_datetime).normalize()
        prev_time = None
        start_times = []

        for t in df["start time"].astype(str).str.strip():
            try:
                time = pd.to_datetime(t, format="%H:%M:%S").time()
            except ValueError:
                # fallback: sometimes single-digit hour/minute
                time = pd.to_datetime(t, format="%H:%M:%S", errors="coerce").time()
            ts = pd.Timestamp.combine(current_date, time)

            # detect day rollover (e.g., 23:59 → 00:00)
            if prev_time and ts < prev_time:
                current_date += pd.Timedelta(days=1)
                ts = pd.Timestamp.combine(current_date, time)

            start_times.append(ts)
            prev_time = ts

        df["Starttime"] = start_times
        df["Duration"] = pd.to_timedelta(df["duration (seconds)"].astype(float), unit="s")
        df["Endtime"] = df["Starttime"] + df["Duration"]
        df["Label"] = df["event"].astype(str).str.strip().str.lower()

        return df[["Label", "Starttime", "Endtime", "Duration"]]