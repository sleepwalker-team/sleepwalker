import os

import pandas as pd
from sleepwalker.datasets.Basedataset import BaseDataset
from sleepwalker.utils import logger
from sleepwalker.datasets.utils import NSRR_EVENT_LABELS, read_profusion
from pathlib import Path

def read_xml(fpath, start_date):
    name = Path(fpath).name.split(".edf")[0]
    subset = Path(fpath).parent.name
    repo_folder = fpath.split("edfs")[0]
    
    xml_path = os.path.join(repo_folder, f"annotations-events-profusion", subset, f"{name}-profusion.xml")
    
    xml_df = read_profusion(xml_path)

    xml_df["Starttime"] = start_date + pd.to_timedelta(pd.to_numeric(xml_df["Start"], errors="raise"), unit="s")
    xml_df["Endtime"] = xml_df["Starttime"] + pd.to_timedelta(pd.to_numeric(xml_df["Duration"], errors="raise"), unit="s")

    columns = ["Label", "Starttime", "Endtime", "SourceLabel"] + (["desaturation_drop_pp"] if "desaturation_drop_pp" in xml_df else [])
    xml_df = xml_df.loc[xml_df["Endtime"] > xml_df["Starttime"], columns].copy()
    
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
            **kwargs
        ): 
        super().__init__(**kwargs)

    def supported_events(self) -> list[str]:
        return list(dict.fromkeys([*NSRR_EVENT_LABELS.values(), "other"]))

    def get_event_df(self, fpath, start_date):
        df = read_xml(fpath, start_date)

        return df
