import os

import pandas as pd
from sleepwalker.datasets.Basedataset import BaseDataset
from sleepwalker.utils import logger
from pathlib import Path
import xmltodict as xtd
from datetime import datetime, timedelta

def read_mnc(xml_path: str, start_date: datetime) -> pd.DataFrame:
    """
    Reads MNC XML sleep annotation files and returns a DataFrame with
    proper absolute timestamps.

    Args:
        xml_path: Path to the XML annotation file.
        start_date: datetime object providing the date (time is ignored).

    Returns:
        DataFrame with columns:
            ['Label', 'Start', 'End', 'Duration', 'Starttime', 'Endtime']
    """
    with open(xml_path, "r") as f:
        doc = xtd.parse(f.read())

    ann = doc.get("Annotations", {})
    instances = ann.get("Instances", {}).get("Instance", [])
    if not isinstance(instances, list):
        instances = [instances]

    df = pd.DataFrame(instances)

    # Normalize label & numeric fields
    df["Label"] = df["@class"].str.lower().str.strip()
    df["Start"] = df["Start"].astype(float)
    df["Duration"] = df["Duration"].astype(float)

    # Parse the local time string, e.g. "22.47.38"
    # Use only hours/minutes/seconds, ignore date from XML
    def parse_local_time(tstr: str) -> datetime:
        return datetime.strptime(tstr, "%H.%M.%S")

    # Build absolute datetimes with rollover handling
    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    prev_time = None
    timestamps = []

    for tstr in df["Name"]:
        t = parse_local_time(tstr)
        # replace date
        dt = current_date.replace(hour=t.hour, minute=t.minute, second=t.second)
        # if time wrapped past midnight
        if prev_time and t < prev_time:
            current_date += timedelta(days=1)
            dt = current_date.replace(hour=t.hour, minute=t.minute, second=t.second)
        timestamps.append(dt)
        prev_time = t

    df["Starttime"] = timestamps
    df["Endtime"] = df["Starttime"] + pd.to_timedelta(df["Duration"], unit="s")

    return df[["Label", "Starttime", "Endtime", "Duration"]]

def read_xml(fpath, trim_wake, start_date):
    name = Path(fpath).name.split(".edf")[0]
    folder = Path(fpath).parent
    
    xml_path = os.path.join(folder, f"{name}.xml")
    xml_df = read_mnc(xml_path, start_date)

    if trim_wake:
        idx = xml_df[xml_df["Label"].isin(["nrem1", "nrem2", "nrem3", "rem"]) ].first_valid_index()
        first_event_timestamp = xml_df.loc[idx]["Starttime"]

        idx = xml_df[xml_df["Label"].isin(["nrem1", "nrem2", "nrem3", "rem"]) ].last_valid_index()
        last_event_timestamp = xml_df.loc[idx]["Starttime"]
        xml_df.loc[xml_df["Endtime"] < (first_event_timestamp - pd.to_timedelta(trim_wake)), "Label"] = None
        xml_df.loc[xml_df["Starttime"] > (last_event_timestamp + pd.to_timedelta(trim_wake)), "Label"] = None
    
    xml_df = xml_df[["Label", "Starttime", "Endtime"]].dropna()

    if len(xml_df) == 0:
        raise ValueError(f"Cannot read XML file for {xml_path}. File is empty?")
    
    return xml_df

class MNC(BaseDataset):
    """
    Dataset URL:
        https://sleepdata.org/datasets/mnc/pages/README.md
        https://sleepdata.org/datasets/mnc/pages/montage-and-sampling-rate-information.md

    Dataset Summary
        Summary of core statistics for this dataset.

        n_patients: 78 
        Durations:
            min     : 0 days 06:30:17
            max     : 0 days 11:09:19
            mean    : 0 days 08:26:08.388489209
            median  : 0 days 08:18:44
            q25     : 0 days 07:51:13
            q75     : 0 days 08:56:42

        Channels:
            1. F3                   100.0%
            2. F4                   100.0%
            3. C3                   100.0%
            4. C4                   100.0%
            5. E2                   100.0%
            6. E1                   100.0%
            7. spo2                 100.0%
            8. abdomen              100.0%
            9. thorax               100.0%
            10. cs_LOC               100.0%
            11. cs_EEG               100.0%
            12. position             100.0%
            13. cs_ROC               100.0%
            14. cs_EMG               100.0%
            15. O1                   99.3%
            16. O2                   99.3%
            17. cs_ECG               95.0%
            18. cchin_l              88.5%
            19. nas_pres             88.5%
            20. flow                 72.7%
            21. ECG1_2               61.2%
            22. rleg                 56.8%
            23. lleg                 56.8%
            24. snore                56.1%
            25. sum                  43.9%
            26. lleg1_2              43.2%
            27. rleg1_2              43.2%
            28. ECG                  27.3%
            29. therm                27.3%
            30. rchin_c              25.9%
            31. chin                 11.5%
            32. ECG2                 11.5%
            33. pulse                7.9%
            34. etco2                3.6%
            35. ECG1                 1.4%
            36. ECG3                 1.4%

        Classes:
            - 
            - 9
            - nrem1
            - nrem2
            - nrem3
            - rem
            - wake

    """

    def __init__(self, 
            trim_wake = "30m",
            ignore_patients_with_partial_events = False,
            **kwargs
        ): 
        
        self.trim_wake = trim_wake
        self.ignore_patients_with_partial_events = ignore_patients_with_partial_events
        super().__init__(**kwargs)

    def has_extra_target(self):
        return False

    def get_event_df(self, fpath, start_date):
        df = read_xml(fpath, self.trim_wake, start_date)

        if self.ignore_patients_with_partial_events and len(self.event_mapping) > 0:
            vals = sorted(set([v for k, v in self.event_mapping.items() if k != "default_class"]))
            labels = sorted(set(df["Label"].unique()))
            if labels != vals:
                raise ValueError(f"Patient did not contain at-least one window with one of the events attached. Patient has {labels}, but expected {vals}.")        
        return df