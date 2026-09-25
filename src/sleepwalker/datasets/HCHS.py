import os

import pandas as pd
from sleepwalker.datasets.Basedataset import BaseDataset
from sleepwalker.utils import logger
from sleepwalker.datasets.utils import read_nsrr, read_profusion
from pathlib import Path

def read_xml(fpath, start_date):
    name = Path(fpath).name.split(".edf")[0]
    repo_folder = fpath.split("edfs")[0]
    
    xml_path = os.path.join(repo_folder, f"annotations-events-nsrr", f"{name}-nsrr.xml")
    xml_df = read_nsrr(xml_path)

    xml_df["Starttime"] = xml_df.apply(lambda row : start_date + pd.to_timedelta(f"{row['Start']} s"),axis=1)
    xml_df["Endtime"] = xml_df.apply(lambda row : start_date + pd.to_timedelta(f"{row['Start']} s" + pd.to_timedelta(f"{row.Duration}s")),axis=1)
    
    xml_df = xml_df[["Label", "Starttime", "Endtime"]].dropna()

    if len(xml_df) == 0:
        raise ValueError(f"Cannot read XML file for {xml_path}. File is empty?")
    
    return xml_df

class HCHS(BaseDataset):
    """
    Dataset Summary
    Summary of core statistics for this dataset.

    n_patients: 12088 
    Durations:
        min     : 0 days 00:40:56
        max     : 0 days 17:12:04
        mean    : 0 days 06:47:08.829500331
        median  : 0 days 07:10:56
        q25     : 0 days 06:00:21
        q75     : 0 days 08:00:05

    Channels:
        1. A-Pulse              100.0%
        2. A-SpO2               100.0%
        3. A-Snore              100.0%
        4. A-Flow               100.0%
        5. HMov                 100.0%
        6. HPos                 100.0%
        7. FVP                  72.6%
        8. A-Effort             27.4%

    Classes:
        - 3 snore + arousal|3 snore + arousal
        - 7 hours completed|7 hours completed
        - a/h tech exc. spo2|a/h tech exc. spo2
        - ae central|ae central
        - ae tech apnea|ae tech apnea
        - ae tech hy0%|ae tech hy0%
        - ae tech hy1%|ae tech hy1%
        - ae tech hy3%|ae tech hy3%
        - ae tech hy4%|ae tech hy4%
        - alarm light|alarm light
        - alarm loud|alarm loud
        - alarm moderate|alarm moderate
        - apnea a-7100-0-0|apnea a-7100-0-0
        - apnea a-7101-0-0|apnea a-7101-0-0
        - apnea a-7103-0-0|apnea a-7103-0-0
        - apnea a-7104-0-0|apnea a-7104-0-0
        - apnea-3506-2-0|apnea-3506-2-0
        - apnea-3507-2-0|apnea-3507-2-0
        - apnea-55-1-0|apnea-55-1-0
        - apnea-55-1-1|apnea-55-1-1
        - apnea-56-1-0|apnea-56-1-0
        - apnea-7110-0-0|apnea-7110-0-0
        - apnea-7111-0-0|apnea-7111-0-0
        - apnea-7114-0-0|apnea-7114-0-0
        - auto low spo2|auto low spo2
        - baseline extra n1|baseline extra n1
        - baseline n1|baseline n1
        - cresc w/arousal|cresc w/arousal
        - de1%|de1%
        - de3%-1101-00|de3%-1101-00
        - de3%-1201-0-0|de3%-1201-0-0
        - de3%-1301-0-0|de3%-1301-0-0
        - de4%|de 4%
        - decrease|decrease
        - deleted apnea a-7200-0-0|deleted apnea a-7200-0-0
        - deleted apnea a-7201-0-0|deleted apnea a-7201-0-0
        - deleted apnea a-7203-0-0|deleted apnea a-7203-0-0
        - deleted apnea a-7204-0-0|deleted apnea a-7204-0-0
        - deleted apnea-7210-0-0|deleted apnea-7210-0-0
        - deleted apnea-7211-0-0|deleted apnea-7211-0-0
        - deleted apnea-7214-0-0|deleted apnea-7214-0-0
        - deleted hypopnea 0% a|deleted hypopnea 0% a
        - deleted hypopnea 1% a|deleted hypopnea 1% a
        - deleted hypopnea 3% a|deleted hypopnea 3% a
        - deleted hypopnea 4% a|deleted hypopnea 4% a
        - deleted hypopnea 4%|deleted hypopnea 4%
        - extib airflow|extib airflow
        - extib awake-5156-0-0|extib awake-5156-0-0
        - extib awake-5156-0-2|extib awake-5156-0-2
        - extib spo2 quality|extib spo2 quality
        - extib tech|extib tech
        - gasp/quiet-3506-0-0|gasp/quiet-3506-0-0
        - gasp/quiet-3507-0-0|gasp/quiet-3507-0-0
        - gross movement|gross movement
        - hrminmax|hrminmax
        - hypopnea 0% a|hypopnea 0% a
        - hypopnea 1% a|hypopnea 1% a
        - hypopnea 3% a|hypopnea 3% a
        - hypopnea 4% a|hypopnea 4% a
        - hypopnea 4%|hypopnea 4%
        - hypopnea-3506-3-0|hypopnea-3506-3-0
        - hypopnea-3507-3-0|hypopnea-3507-3-0
        - hypopnea-55-2-0|hypopnea-55-2-0
        - hypopnea-55-2-1|hypopnea-55-2-1
        - hypopnea-56-2-0|hypopnea-56-2-0
        - increase-55-0-0|increase-55-0-0
        - increase-55-0-1|increase-55-0-1
        - increase-56-0-0|increase-56-0-0
        - increase|increase
        - low battery|low battery
        - movement arousal|movement arousal
        - quality-bad|quality-bad
        - quality-marginal|quality-marginal
        - quality-poor|quality-poor
        - start n1|start n1
        - start tib n1|start tib n1
        - stop tib n1|stop tib n1
        - tech exc. spo2|tech exc. spo2
        - tech exclude snoring|tech exclude snoring
        - unknown event 13|unknown event 13
        - unknown event 17|unknown event 17
        - unknown event 18|unknown event 18
        - unknown event 19|unknown event 19
        - unknown event 1|unknown event 1
        - unknown event 20|unknown event 20
        - unknown event 21|unknown event 21
        - unknown event 22|unknown event 22
        - unknown event 23|unknown event 23
        - unknown event 24|unknown event 24
        - unknown event 25|unknown event 25
        - unknown event 26|unknown event 26
        - unknown event 27|unknown event 27
        - unknown event 28|unknown event 28
        - unknown event 29|unknown event 29
        - unknown event 2|unknown event 2
        - unknown event 30|unknown event 30
        - unknown event 31|unknown event 31
        - unknown event 32|unknown event 32
        - unknown event 33|unknown event 33
        - unknown event 34|unknown event 34
        - unknown event 35|unknown event 35
        - unknown event 36|unknown event 36
        - unknown event 37|unknown event 37
        - unknown event 38|unknown event 38
        - unknown event 39|unknown event 39
        - unknown event 3|unknown event 3
        - unknown event 40|unknown event 40
        - unknown event 41|unknown event 41
        - unknown event 42|unknown event 42
        - unknown event 43|unknown event 43
        - unknown event 45|unknown event 45
        - unknown event 4|unknown event 4
        - user shutoff|user shutoff
    """


    def __init__(self, 
            ignore_patients_with_partial_events = False,
            **kwargs
        ): 
        
        self.ignore_patients_with_partial_events = ignore_patients_with_partial_events
        super().__init__(**kwargs)

    def has_extra_target(self):
        return False

    def get_event_df(self, fpath, start_date):
        df = read_xml(fpath, start_date)

        if self.ignore_patients_with_partial_events and len(self.event_mapping) > 0:
            vals = sorted(set([v for k, v in self.event_mapping.items() if k != "default_class"]))
            labels = sorted(set(df["Label"].unique()))
            if labels != vals:
                raise ValueError(f"Patient did not contain at-least one window with one of the events attached. Patient has {labels}, but expected {vals}.")        
        return df