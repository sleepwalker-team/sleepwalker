from __future__ import annotations
from datetime import datetime, timedelta

import pandas as pd
from sleepwalker.utils import logger
from sleepwalker.datasets.Basedataset import BaseDataset

def convert_to_datetime(row, start_date):
    try:
        row_ts = datetime.strptime(row, '%H:%M:%S')
    except:
        row_ts = datetime.strptime(row, '%H.%M.%S')
    hours, minutes, seconds = row_ts.hour, row_ts.minute, row_ts.second
    year, month, day = start_date.year, start_date.month, start_date.day
    complete_date = datetime(year, month, day, hours, minutes, seconds)
    if hours < 10:
        # Kind of hacky: Whenever an hour entry between 00:00 - 10:00 occurs, we assume that this is the next day 
        # and add one day to the complete_date
        complete_date += timedelta(days=1)

    return complete_date

class CAP(BaseDataset):
    # Some files are corrupted / wrong metadata. Interestingly, this implementation partially supports a different subset compared to the literature. For example, we can load brux1 without any issue due to our re-sampling. However, to make things a bit more comparable, we will exclude the same patients here as it has been done in the literature. 
    # See: 
    #   - U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging by Perslev et al. 2019
    #   - Multichannel Sleep Stage Classification and Transfer Learning using Convolutional Neural Networks by Andreotti et al. 2019 
    FILES_TO_EXCLUDE = ['brux1.edf', 'n4.edf', 'n8.edf', 'n16.edf', 'nfle6.edf', 'nfle25.edf', 'nfle27.edf', 'nfle33.edf', 'n12.edf', 'n16.edf']

    def __init__(self, 
            trim_wake = "30m",
            disorders = ["n", "ins", "narco", "nfle", "rbd", "sdb", "brux", "plm"],
            **kwargs
        ): 
        
        self.trim_wake = trim_wake
        self.disorders = disorders

        if self.disorders is None:
            self.disorders = ["n", "ins", "narco", "nfle", "rbd", "sdb", "brux", "plm"]

        for d in self.disorders:
            if d not in ["n", "ins", "narco", "nfle", "rbd", "sdb", "brux", "plm"]:
                logger.warining(f"Unknown disorder found in CAP configuration. The original CAP dataset does not contain a disorder `{d}`. Typically, only {{n, ins, narco, nfle, rbd, sdb, brux, plm}} are available. If you do not want to filter for specific disorders or filenames have changed on disk, you can ignore this warning.")

        super().__init__(**kwargs)
        
    def has_extra_target(self):
        return False

    def get_event_df(self, fpath, start_date):
        label_path = fpath.replace('edf', 'txt')

        # The first couple of lines of the document are metadata. Skip
        to_skip = 0
        date = None
        with open(label_path, 'r') as f:
            while True:
                line = f.readline()
                if "Recording Date" in line:
                    date = datetime.strptime(line.split("Recording Date:\t")[1].strip(), "%m/%d/%Y")
                if line.startswith('Sleep Stage'):
                    break
                else:
                    to_skip += 1

        df = pd.read_csv(label_path, skiprows=to_skip, sep='\t').dropna()
        if "sdb1.edf" in fpath:
            # "Recording Date" denotes the date when the recording started and SDB1.edf started after 0:00 h
            # Hence, convert_to_datetime will add 1 day to date which is wrong, since Recording Date is already correct in this case
            date -= timedelta(days=1)

        # if "n12.edf" in fpath:
        #     # The dates in the EDF file do not match the dates in the txt file by exactly one year. Maybe a typo?
        #     date -= timedelta(days=365)
        
        df['Starttime'] = df['Time [hh:mm:ss]'].apply(lambda row: convert_to_datetime(row, date))
        
        if "Duration[s]" in df.columns:
            df['Endtime'] = df['Starttime'] + df['Duration[s]'].apply(lambda row: pd.to_timedelta(f'{row} s'))
        else:
            df['Endtime'] = df['Starttime'] + df['Duration [s]'].apply(lambda row: pd.to_timedelta(f'{row} s'))
        df['Label'] = df['Sleep Stage']
        
        if self.trim_wake:
            sleep_stages = ["W", "S1", "S2", "S3", "W", "R"]
            idx = df[df["Label"].isin(sleep_stages) ].first_valid_index()
            first_event_timestamp = df.loc[idx]["Starttime"]

            idx = df[df["Label"].isin(sleep_stages) ].last_valid_index()
            last_event_timestamp = df.loc[idx]["Starttime"]
            df.loc[df["Endtime"] < (first_event_timestamp - pd.to_timedelta(self.trim_wake)), "Label"] = None
            df.loc[df["Starttime"] > (last_event_timestamp + pd.to_timedelta(self.trim_wake)), "Label"] = None

        # df["Label"] = df.apply(lambda row: None if row.Label is None else row.Label.lower().strip(), axis=1)
        # df["Label"] = df.apply(lambda row: self.event_mapping[row["Label"]] if row["Label"] in self.event_mapping else None, axis=1)
        df = df[['Label', 'Starttime', 'Endtime']].dropna()

        return df