import os

import pandas as pd
from sleepwalker.datasets.Basedataset import BaseDataset
from sleepwalker.utils import logger
from sleepwalker.datasets.utils import read_profusion
from pathlib import Path

def read_xml(fpath, trim_wake, start_date):
    name = Path(fpath).name.split(".edf")[0]
    subset = Path(fpath).parent.name
    repo_folder = fpath.split("edfs")[0]
    
    xml_path = os.path.join(repo_folder, f"annotations-events-profusion", subset, f"{name}-profusion.xml")
    
    xml_df = read_profusion(xml_path)

    xml_df["Starttime"] = xml_df.apply(lambda row : start_date + pd.to_timedelta(f"{row['Start']} s"),axis=1)
    xml_df["Endtime"] = xml_df.apply(lambda row : start_date + pd.to_timedelta(f"{row['Start']} s" + pd.to_timedelta(f"{row.Duration}s")),axis=1)

    if trim_wake:
        idx = xml_df[xml_df["Label"].isin(["stage 1 sleep|1", "stage 2 sleep|2", "stage 3 sleep|3", "stage 4 sleep|4", "rem sleep|5"]) ].first_valid_index()
        first_event_timestamp = xml_df.loc[idx]["Starttime"]

        idx = xml_df[xml_df["Label"].isin(["stage 1 sleep|1", "stage 2 sleep|2", "stage 3 sleep|3", "stage 4 sleep|4", "rem sleep|5"]) ].last_valid_index()
        last_event_timestamp = xml_df.loc[idx]["Starttime"]
        xml_df.loc[xml_df["Endtime"] < (first_event_timestamp - pd.to_timedelta(trim_wake)), "Label"] = None
        xml_df.loc[xml_df["Starttime"] > (last_event_timestamp + pd.to_timedelta(trim_wake)), "Label"] = None
    
    xml_df = xml_df[["Label", "Starttime", "Endtime"]].dropna()

    if len(xml_df) == 0:
        raise ValueError(f"Cannot read XML file for {xml_path}. File is empty?")
    
    return xml_df

class Numom2b(BaseDataset):
    """Dataset Summary
        Summary of core statistics for this dataset.

        n_patients: 1206 
        Durations:
            min     : 0 days 02:42:10
            max     : 3 days 11:27:00
            mean    : 0 days 11:52:43.208955224
            median  : 0 days 12:00:10
            q25     : 0 days 12:00:10
            q75     : 0 days 12:00:10

        Channels:
            1. nas_pres             100.0%
            2. ECG                  100.0%
            3. thorax               100.0%
            4. snore                100.0%
            5. spo2                 100.0%
            6. activity             100.0%
            7. pleth                100.0%
            8. position             100.0%
            9. DHR                  100.0%
            10. pulse                99.9%
            11. abdomen              99.9%
            12. sum                  99.8%
            13. flow                 55.7%
            14. xsum                 0.3%
            15. therm                0.2%

        Classes:
            - central apnea|central apnea
            - hypopnea|hypopnea
            - mixed apnea|mixed apnea
            - obstructive apnea|obstructive apnea
            - periodic breathing|periodic breathing
            - respiratory artifact|respiratory artifact
            - spo2 artifact|spo2 artifact
            - spo2 desaturation|spo2 desaturation
            - stage 2 sleep|2
            - stage 3 sleep|3
            - unscored|9
            - unsure|unsure
            - wake|0
    """
    def __init__(self, 
            trim_wake = "30m",
            ignore_patients_with_partial_events = False,
            **kwargs
        ): 
        
        self.trim_wake = trim_wake
        self.ignore_patients_with_partial_events = ignore_patients_with_partial_events
        super().__init__(**kwargs)

    def get_event_df(self, fpath, start_date):
        df = read_xml(fpath, self.trim_wake, start_date)

        if self.ignore_patients_with_partial_events and len(self.event_mapping) > 0:
            vals = sorted(set([v for k, v in self.event_mapping.items() if k != "default_class"]))
            labels = sorted(set(df["Label"].unique()))
            if labels != vals:
                raise ValueError(f"Patient did not contain at-least one window with one of the events attached. Patient has {labels}, but expected {vals}.")        
        return df