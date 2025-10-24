from datetime import datetime, timedelta
import pandas as pd
import os
from sleepwalker.datasets.Basedataset import BaseDataset
from datetime import timedelta

def convert_to_datetime(row, start_date):
    row_ts = datetime.strptime(row, '%H:%M:%S')
    hours, minutes, seconds = row_ts.hour, row_ts.minute, row_ts.second
    year, month, day = start_date.year, start_date.month, start_date.day
    complete_date = datetime(year, month, day, hours, minutes, seconds)
    if complete_date <= start_date:
        complete_date += timedelta(days=1)

    return complete_date

class SVUH_UCD(BaseDataset):
    """Dataset Summary
        Summary of core statistics for this dataset.

        n_patients: 25 
        Durations:
            min     : 0 days 05:55:50
            max     : 0 days 07:42:39
            mean    : 0 days 06:56:03.920000
            median  : 0 days 06:53:18
            q25     : 0 days 06:34:00
            q75     : 0 days 07:26:31

        Channels:
            1. Lefteye              100.0%
            2. RightEye             100.0%
            3. EMG                  100.0%
            4. C3A2                 100.0%
            5. C4A1                 100.0%
            6. ECG                  100.0%
            7. SpO2                 100.0%
            8. Flow                 100.0%
            9. Sum                  100.0%
            10. Pulse                100.0%
            11. ribcage              100.0%
            12. abdo                 100.0%
            13. BodyPos              100.0%
            14. Sound                60.0%
            15. Soud                 40.0%
            16. Right leg            4.0%
            17. Left leg             4.0%

        Classes:
            - 0
            - 1
            - 2
            - 3
            - 4
            - 5
            - 8
            - apnea-
            - apnea-c
            - apnea-m
            - apnea-o
            - hyp-c
            - hyp-m
            - hyp-o
            - pb even
            - possib
            - possibl
            - possible
    """
    def __init__(self, 
            end_align = False,
            **kwargs
        ): 
        self.end_align = end_align
        super().__init__(**kwargs)

    def has_extra_target(self):
        return False

    def get_event_df(self, fpath, start_date):
        repo_folder, filename = os.path.split(fpath) 
        pid = filename.split('.rec')[0]
        metadata = pd.read_excel(os.path.join(repo_folder, 'SubjectDetails.xls'))
        metadata['Study Number'] = metadata['Study Number'].str.lower()

        # Load and map corresponding sleep labels
        with open(os.path.join(repo_folder, f'{pid}_stage.txt'), 'r') as f:
            sleep_stages = [line for line in f.read().splitlines()]

        if self.end_align:
            # Assume that scoring ends at the end of the PSG measurements
            edf_duration = metadata[self.metadata['Study Number'] == pid]['No of data blocks in EDF'].item()
            txt_duration = len(sleep_stages)*30
            offset = max(edf_duration - txt_duration, 0)
            start_date = start_date + timedelta(seconds=offset)

        df_sleep = pd.DataFrame(columns=['Label', 'Starttime', 'Endtime'])
        df_sleep['Label'] = sleep_stages
        df_sleep['Starttime'] = [start_date + pd.to_timedelta(f'{i * 30} s') for i in range(len(sleep_stages))]
        df_sleep['Endtime'] = [start_date + pd.to_timedelta('30 s') + pd.to_timedelta(f'{i * 30} s') for i in range(len(sleep_stages))]

        # Now, also add the other respiratory events
        resp_events = pd.read_fwf(os.path.join(repo_folder, f'{pid}_respevt.txt'), infer_nrows=10, skiprows=2, names=['time', 'type', 'pb/cs', 'duration', 'desat_low', 'desat_drop', 'snore', 'arousal', 'B/T rate', 'B/T change'], header=0)
        resp_events = resp_events.dropna(subset=['duration'])
        resp_events['time'] = resp_events['time'].apply(lambda row: convert_to_datetime(row, start_date))
        df_resp = pd.DataFrame(columns=['Label', 'Starttime', 'Endtime'])
        df_resp['Label'] = resp_events['type'].str.lower()
        df_resp['Starttime'] = resp_events['time'].apply(pd.to_datetime)
        df_resp['Endtime'] = resp_events['time'].apply(pd.to_datetime) + resp_events['duration'].apply(lambda x: pd.to_timedelta(f'{x} s'))

        # Theoretically, we could get some readings of Arousal, Desaturation, Brachycardia and Tachycardia out of this, but they are very coarse and strange.
        # It seams that sleep stages and respiratory events are the best annotated ones

        df = pd.concat([df_sleep, df_resp])

        df["Label"] = df.apply(lambda row: None if row.Label is None else row.Label.lower().strip(), axis=1)
        df = df.dropna()
        return df
