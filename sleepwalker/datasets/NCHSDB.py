from __future__ import annotations
from datetime import datetime, timedelta

import pandas as pd
from .Basedataset import BaseDataset  


class NCHSDB(BaseDataset):
    def __init__(self, label_should_contain = "Sleep stage", **kwargs):
        self.label_should_contain = label_should_contain
        super().__init__(**kwargs)

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        """
        Load TSV annotations for the NCHSDB dataset.

        Args:
            edf_path: Path to the EDF file (used to locate .tsv annotation file)
            start_datetime: Timestamp indicating when the EDF recording started

        Returns:
            pd.DataFrame with columns:
                ['Label', 'Starttime', 'Endtime', 'Duration']
        """
        annot_path = f"{edf_path.split('.edf')[0]}.tsv"
        df = pd.read_csv(annot_path, sep="\t")

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Compute proper datetime columns
        # onset and duration are in seconds from start_datetime
        start_times = []
        end_times = []

        # current_date = start_datetime.normalize()
        prev_time = start_datetime.normalize()

        for onset, dur in zip(df["onset"], df["duration"]):
            start_ts = start_datetime + pd.to_timedelta(onset, unit="s")
            end_ts = start_ts + pd.to_timedelta(dur, unit="s")

            # Rollover handling: if we cross midnight
            if start_ts.time() < prev_time.time() and (start_ts - prev_time) < pd.Timedelta("12h"):
                start_ts += pd.Timedelta(days=1)
                end_ts += pd.Timedelta(days=1)
            prev_time = start_ts

            start_times.append(start_ts)
            end_times.append(end_ts)

        df["Label"] = df["description"].astype(str).str.strip()
        df["Starttime"] = start_times
        df["Endtime"] = end_times
        df["Duration"] = pd.to_timedelta(df["duration"], unit="s")

        if self.label_should_contain is not None and self.label_should_contain != "":
            df.loc[~df["Label"].str.contains(self.label_should_contain, na=False), :] = None
            df = df.dropna()

        return df[["Label", "Starttime", "Endtime", "Duration"]]