from __future__ import annotations

import os
import traceback
from sleepwalker.utils import logger

import pandas as pd
import xlrd

from sleepwalker.datasets.Basedataset import BaseDataset

class Ruhrlandklinik(BaseDataset):
    def __init__(self, 
            return_nox = True,
            cutoff_time = "adaptive",
            merge_overlapping_lm = True,
            round_lm_up = "11s",
            round_lm_down = "0.49s",
            drop_patients_wrong_lm = "False",
            fix_awake = True,
            **kwargs
        ): 

        self.return_nox = return_nox
        self.cutoff_time = cutoff_time
        self.merge_overlapping_lm = merge_overlapping_lm
        self.round_lm_up = round_lm_up
        self.round_lm_down = round_lm_down
        self.drop_patients_wrong_lm = drop_patients_wrong_lm
        self.fix_awake = fix_awake
        super().__init__(**kwargs)

    def get_extra_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        nox_filepath = os.path.splitext(edf_path)[0] + "_AS.csv"
        try:
            dfnox = pd.read_csv(nox_filepath, skiprows=[1], header=0, parse_dates=["Anfangszeit","Endzeit"], dayfirst=True, usecols=[0,1,2,3])
            
            # For some reason the encoding of the CSV seems to be broken already on disk? Hence loading with a different encoding does not really change much
            # Luckily, only a few Umlaute are broken which we can fix here directly
            encoding_fix = {
                "r?ckenlage":"rückenlage",
                "ents?ttigung":"entsättigung",
                "ger?teeinstellungen":"geräteeinstellungen"
            }
            dfnox["Ereignis"] = dfnox.apply(lambda row: encoding_fix.get(row.Ereignis.lower().strip(), row.Ereignis.lower().strip()), axis=1) # type: ignore
            # dfnox["Ereignis"] = dfnox.apply(lambda row: self.event_mapping[row["Ereignis"]] if row["Ereignis"] in self.event_mapping else None, axis=1) # type: ignore
            dfnox = dfnox.rename(columns={"Ereignis":"Label", "Anfangszeit":"Starttime", "Endzeit":"Endtime", "Dauer":"Duration"})
            dfnox = dfnox.dropna()
            dfnox["Starttime"] = pd.to_datetime(dfnox["Starttime"])
            dfnox["Endtime"] = pd.to_datetime(dfnox["Endtime"])

            return dfnox
        except Exception as e:
                raise ValueError(f"Error reading {nox_filepath}. Error was {traceback.format_exc()}")
    
    def has_extra_target(self):
        return self.return_nox

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        description_filepath = os.path.splitext(edf_path)[0] + "_MS"
        if os.path.exists(description_filepath + ".xls"):
            wb = xlrd.open_workbook(description_filepath + ".xls", logfile=open(os.devnull, 'w'))
            xls_file = pd.read_excel(wb, engine='xlrd')
        elif os.path.exists(description_filepath + ".xlsx"):
            wb = xlrd.open_workbook(description_filepath + ".xlsx", logfile=open(os.devnull, 'w'))
            xls_file = pd.read_excel(wb, engine='xlrd')
        else:
            raise ValueError(f"XLS/XLSX description file does not exist for input file '{edf_path}'.")

        # Drop first row of XLS file, which contains meta information.
        xls_file = xls_file[1:]
        xls_file["Ereignis"] = xls_file["Ereignis"].str.lower()

        xls_file["Anfangszeit"] = pd.to_datetime(xls_file["Anfangszeit"])
        xls_file["Endzeit"] = pd.to_datetime(xls_file["Endzeit"])

        # TODO 
        # Retrieve event time boundaries.
        if self.cutoff_time == "adaptive":
            # Usually we would assume that people would be tagging the start/end of their analysis by a "Start der Analyse" tag. However, only a fraction (8/304 during development) patients had these tags. The tagging of sleepstages seems more reliable. Usually I would assume that a patient start in "Wake" and ends in "Wake", however, this also seems somewhat unreliable. Hence, we take any sleep stage here as boundary
            idx = xls_file[xls_file["Ereignis"].isin(["wach", "n1", "n2", "n3", "rem"]) ].first_valid_index()
            first_event_timestamp = xls_file.loc[idx]["Anfangszeit"]
            
            idx = xls_file[xls_file["Ereignis"].isin(["wach", "n1", "n2", "n3", "rem"]) ].last_valid_index()
            last_event_timestamp = xls_file.loc[idx]["Endzeit"]
        else:
            first_event_timestamp = xls_file.iloc[0]["Anfangszeit"] + pd.to_timedelta(self.cutoff_time) #datetime.timedelta(minutes=cutoff_minutes)
            last_event_timestamp = xls_file.iloc[-1]["Endzeit"] - pd.to_timedelta(self.cutoff_time) #datetime.timedelta(minutes=cutoff_minutes)

        events = list(set(self.event_mapping.keys()))
        if "LM" in events and self.merge_overlapping_lm:
            dff = xls_file.loc[xls_file["Ereignis"] == "lm",:]
            #dff['Dauer'] = dff["Dauer"].astype(np.float64) #dff.loc[:,'Dauer'].apply(pd.to_numeric)
            merged_rows = []
            
            current_start = dff["Anfangszeit"].values[0]
            current_end = dff["Endzeit"].values[0]

            for i in range(1, len(dff)):
                if dff["Anfangszeit"].values[i] <= current_end:  # If the start time overlaps
                    current_end = max(current_end, dff["Endzeit"].values[i])  # Extend the end time if necessary
                else:
                    merged_rows.append({"Anfangszeit": current_start, "Endzeit": current_end, "Ereignis": "lm", "Dauer": (current_end - current_start).total_seconds()})
                    current_start = dff["Anfangszeit"].values[i]
                    current_end = dff["Endzeit"].values[i]
            merged_rows.append({"Anfangszeit": current_start, "Endzeit": current_end, "Ereignis": "lm", "Dauer": (current_end - current_start).total_seconds()})
            dff = pd.DataFrame(merged_rows)
            xls_file = xls_file.loc[xls_file["Ereignis"] != "lm",:]
            xls_file = pd.concat([xls_file, dff], axis=0)

        non_sleep_events = [e for e in events if e not in ["wach", "n1", "n2", "n3", "rem"]]
        non_cnt = 0

        if self.fix_awake and len(non_sleep_events) > 0:
            # Filter for "Wach" events and merge consecutive intervals
            wach_df = xls_file[xls_file["Ereignis"] == "wach"].sort_values("Anfangszeit").reset_index(drop=True)

            # Merge consecutive rows where Endzeit of one row is the same as Anfangszeit of the next.
            merged_wach_intervals = []
            current_start = wach_df.loc[0, "Anfangszeit"]
            current_end = wach_df.loc[0, "Endzeit"]

            for i in range(1, len(wach_df)):
                row = wach_df.iloc[i]
                if row["Anfangszeit"] <= current_end:  
                    # Extend the interval
                    current_end = max(current_end, row["Endzeit"])  
                else:
                    # Save the completed interval + Start a new interval
                    merged_wach_intervals.append((current_start, current_end))
                    current_start = row["Anfangszeit"]
                    current_end = row["Endzeit"]

            # Append the last interval
            merged_wach_intervals.append((current_start, current_end))
            merged_wach_df = pd.DataFrame(merged_wach_intervals, columns=["Anfangszeit", "Endzeit"])
            other_df = xls_file[xls_file["Ereignis"].isin(non_sleep_events)]

            # Check for overlaps and set rows to None where there's an overlap with merged "Wach" intervals
            for idx, event_row in other_df.iterrows():
                event_start = event_row["Anfangszeit"]
                event_end = event_row["Endzeit"]
                
                # Find if there's an overlap with any merged "Wach" interval
                overlap_exists = merged_wach_df[
                    (merged_wach_df["Anfangszeit"] <= event_end) & (merged_wach_df["Endzeit"] >= event_start)
                ]
                
                # If overlap exists, set "Ereignis" of the "Wach" row to None
                if not overlap_exists.empty:
                    non_cnt += 1
                    xls_file.at[idx, "Ereignis"] = None
        
        if non_cnt > 0 and self.verbose in ["TQDM", "tqdm", "console"]:
            logger.info(f"Removed {non_cnt} annotations for patient {edf_path}, due to not being asleep.")

        arousals = ["arousal", "plm-arousal", "rera"]
        if any(a.lower() in events for a in arousals):
            # Remove all arousals that are either too short or too long. 
            # Quantiles for the data distribution (December 2024) were (0, 0.25, 0.5, 0.75, 1): 
            # [0.0 2.21 3.0 3.0 4.818 9.0 15.0 25.52122 125.57799999999999]
            xls_file.loc[(xls_file["Ereignis"].isin(arousals)) & (xls_file["Dauer"] < 1), "Ereignis"] = None
            xls_file.loc[(xls_file["Ereignis"].isin(arousals)) & (xls_file["Dauer"] > 60), "Ereignis"] = None

        apneas = ["a. gemischt", "a. obstruktiv", "a. zentral", "apnoe", "cheyne stokes", "fg apnoe geschlossen", "fg apnoe geöffnet", "fg hypopnoe", "fg-apnoe unbekannt", "h. obstruktiv", "h. zentral", "hypopnea-gemischt", "hypopnoe"]

        if any(a.lower() in events for a in apneas):
            # Remove all apneas that are either too short or too long. 
            # Quantiles for the data distribution (November 2024) were (0, 0.25, 0.5, 0.75, 1): 
            # [0.0 11.0 13.520999999999999 17.494999999999997 23.2845 30.759999999999998 119.493]
            xls_file.loc[(xls_file["Ereignis"].isin(apneas)) & (xls_file["Dauer"] < 5), "Ereignis"] = None
            xls_file.loc[(xls_file["Ereignis"].isin(apneas)) & (xls_file["Dauer"] > 60), "Ereignis"] = None

        lms = ["PLM"]
        if any(a.lower() in events for a in lms):
            # Remove all apneas that are either too short or too long. 
            # Quantiles for the data distribution (November 2024): 
            # 0.699 20.36125 90.1345 139.50074999999998 233.75549999999998 470.84799999999996 1116.8415 5450.110000000001 16960.272]
            xls_file.loc[(xls_file["Ereignis"].isin(lms)) & (xls_file["Dauer"] < 0.7), "Ereignis"] = None
            #xls_file.loc[(xls_file["Ereignis"].isin(lms)) & (xls_file["Dauer"] > 250), "Ereignis"] = None

        xls_file["Ereignis"] = xls_file["Ereignis"].apply(lambda x: None if x is None else x.lower().strip()).dropna()
        df = xls_file[["Ereignis", "Anfangszeit", "Endzeit", "Dauer"]].copy().rename(columns={"Ereignis":"Label", "Anfangszeit":"Starttime", "Endzeit":"Endtime", "Dauer":"Duration"})
        return df 
        # xls_file["Ereignis"] = xls_file["Ereignis"].apply(lambda x: self.event_mapping[x] if x in self.event_mapping else None)

        # xls_file["Ereignis"] = xls_file.apply(lambda row: None if row.Ereignis is None else row.Ereignis.lower().strip(), axis=1)
        # xls_file["Ereignis"] = xls_file.apply(lambda row: self.event_mapping[row["Ereignis"]] if row["Ereignis"] in self.event_mapping else None, axis=1)
        # df = xls_file[["Ereignis", "Anfangszeit", "Endzeit", "Dauer"]].copy().rename(columns={"Ereignis":"Label", "Anfangszeit":"Starttime", "Endzeit":"Endtime", "Dauer":"Duration"})
        # df = df.dropna()

        # return df
