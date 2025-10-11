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

