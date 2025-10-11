import os

import pandas as pd
from sleepwalker.datasets.Basedataset import BaseDataset
from sleepwalker.utils import logger
from sleepwalker.datasets.utils import read_nsrr, read_profusion

def read_xml(fpath, annotator, trim_wake, start_date):
    name = os.path.basename(fpath).split(".edf")[0]
    subset = name.split("-")[0]

    repo_folder = fpath.split("edfs")[0]
    
    xml_path = os.path.join(repo_folder, f"annotations-events-{annotator}", subset, f"{name}-{annotator}.xml")
    if annotator == "nsrr":
        xml_df = read_nsrr(xml_path)
    else:
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

class SHHS(BaseDataset):
    def __init__(self, 
            annotator = "nsrr",
            trim_wake = "30m",
            ignore_patients_with_partial_events = False,
            **kwargs
        ): 
        
        if annotator not in ["nsrr", "profusion"]:
            raise ValueError(f"Unknown value for annotator given. Received {annotator}, but expected {{nsrr, profusion}}")
        else:
            self.annotator = annotator

        self.trim_wake = trim_wake
        self.ignore_patients_with_partial_events = ignore_patients_with_partial_events
        super().__init__(**kwargs)

    def has_extra_target(self):
        return True

    def get_extra_event_df(self, fpath, start_date):
        if self.annotator == "profusion":
            df = read_xml(fpath, "nsrr", self.trim_wake, start_date)
        else:
            df = read_xml(fpath, "profusion", self.trim_wake, start_date)

        return df 

    def get_event_df(self, fpath, start_date):
        df = read_xml(fpath, self.annotator, self.trim_wake, start_date)

        if self.ignore_patients_with_partial_events and len(self.event_mapping) > 0:
            vals = sorted(set([v for k, v in self.event_mapping.items() if k != "default_class"]))
            labels = sorted(set(df["Label"].unique()))
            if labels != vals:
                raise ValueError(f"Patient did not contain at-least one window with one of the events attached. Patient has {labels}, but expected {vals}.")        
        return df