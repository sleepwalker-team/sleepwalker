from __future__ import annotations
from datetime import datetime, timedelta

import pandas as pd
from .Basedataset import BaseDataset  

def convert_to_datetime(row, start_date):
    fmt = "%H:%M:%S.%f" if "." in row else "%H:%M:%S"
    row_ts = datetime.strptime(row, fmt)
    # try:
    # except:
    #     row_ts = datetime.strptime(row, '%H.%M.%S')
    hours, minutes, seconds, microseconds = row_ts.hour, row_ts.minute, row_ts.second, row_ts.microsecond
    year, month, day = start_date.year, start_date.month, start_date.day
    complete_date = datetime(year, month, day, hours, minutes, seconds, microseconds)
    if hours < 10:
        # Kind of hacky: Whenever an hour entry between 00:00 - 10:00 occurs, we assume that this is the next day 
        # and add one day to the complete_date
        complete_date += timedelta(days=1)

    return complete_date

class Apples(BaseDataset):
    """
    Dataset URL: 
        https://sleepdata.org/datasets/apples/pages/README.md
        https://gitlab-scm.partners.org/zzz-public/nsrr/-/tree/master/studies/apples

    Dataset Summary
        Summary of core statistics for this dataset.

        n_patients: 1084 
        Durations:
            min     : 0 days 00:03:12
            max     : 0 days 11:15:00
            mean    : 0 days 08:10:24.836956522
            median  : 0 days 08:13:30
            q25     : 0 days 07:56:30
            q75     : 0 days 08:40:00

        Channels:
            1. snore                100.0%
            2. SpO2                 100.0%
            3. LOC                  100.0%
            4. EMG                  100.0%
            5. ECG                  99.9%
            6. O2_M1                99.4%
            7. C4_M1                99.4%
            8. C3_M2                99.4%
            9. thorax               99.4%
            10. thermistor           99.4%
            11. abdomen              99.4%
            12. nasal_pres           99.4%
            13. ROC                  99.4%
            14. O1_M2                99.2%
            15. LEG                  79.8%
            16. pulse                70.1%

        Classes:
            - L
            - LM
            - N1
            - N2
            - N3
            - R
            - W
            - apnea
            - arousal
            - desat
            - epos2_mixed
            - epos2_nonsupine
            - epos2_supine
            - epos5_left
            - epos5_mixed
            - epos5_prone
            - epos5_right
            - epos5_supine
            - epos5_upright
            - epos9_left
            - epos9_mixed
            - epos9_prone
            - epos9_prone_left
            - epos9_prone_right
            - epos9_right
            - epos9_supine
            - epos9_supine_left
            - epos9_supine_right
            - epos9_upright
            - hypopnea
            - pos2_nonsupine
            - pos2_supine
            - pos5_left
            - pos5_prone
            - pos5_right
            - pos5_supine
            - pos5_upright
            - pos9_left
            - pos9_prone
            - pos9_prone_left
            - pos9_prone_right
            - pos9_right
            - pos9_supine
            - pos9_supine_left
            - pos9_supine_right
            - pos9_upright
            - snoring
        """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        annot_path = f"{edf_path.split('.edf')[0]}.annot"

        df = pd.read_csv(annot_path, sep="\t", dtype={"class":str, "instance":str, "channel":str, "meta":str})
        df = df.rename(columns={"class":"Label", "start":"Starttime", "stop":"Endtime"})

        df['Starttime'] = df['Starttime'].apply(lambda row: convert_to_datetime(row, start_datetime))
        df['Endtime'] = df['Endtime'].apply(lambda row: convert_to_datetime(row, start_datetime))

        df["Duration"] = df["Endtime"] - df["Starttime"]

        return df[["Label", "Starttime", "Endtime", "Duration"]]

