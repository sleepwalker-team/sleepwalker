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
    """
    Dataset URL:
        https://physionet.org/content/capslpdb/1.0.0/

    Dataset Summary
        Summary of core statistics for this dataset.

        n_patients: 106 
        Durations:
            min     : 0 days 03:59:01
            max     : 0 days 15:10:00
            mean    : 0 days 09:17:45.333333333
            median  : 0 days 08:47:03
            q25     : 0 days 08:23:15.750000
            q75     : 0 days 09:35:07.500000

        Channels:
            1. F4-C4                91.7%
            2. P4-O2                91.7%
            3. C4-P4                91.7%
            4. C4-A1                91.7%
            5. ROC-LOC              88.9%
            6. EMG1-EMG2            88.9%
            7. ECG1-ECG2            88.9%
            8. SX1-SX2              82.4%
            9. Fp2-F4               81.5%
            10. HR                   81.5%
            11. DX1-DX2              79.6%
            12. SAO2                 78.7%
            13. PLETH                71.3%
            14. STAT                 68.5%
            15. C3-P3                64.8%
            16. F3-C3                64.8%
            17. P3-O1                64.8%
            18. FP1-F3               63.9%
            19. F8-T4                63.0%
            20. F7-T3                63.0%
            21. T4-T6                41.7%
            22. T3-T5                41.7%
            23. TORACE               28.7%
            24. MIC                  25.0%
            25. ADDOME               20.4%
            26. Position             11.1%
            27. Ox Status            7.4%
            28. FP2-F4               7.4%
            29. Pleth                7.4%
            30. Flusso               5.6%
            31. ADDDOME              5.6%
            32. ECG                  4.6%
            33. C3-A2                4.6%
            34. TIB Dx               3.7%
            35. LOC                  3.7%
            36. ROC                  3.7%
            37. TIB Sx               3.7%
            38. O2-A1                3.7%
            39. F2-F4                2.8%
            40. Dx1-DX2              2.8%
            41. SpO2                 2.8%
            42. T4                   2.8%
            43. LOC-ROC              2.8%
            44. TAG                  2.8%
            45. THE                  2.8%
            46. TERMISTORE           2.8%
            47. LOC-A1               2.8%
            48. EKG                  2.8%
            49. ROC-A2               2.8%
            50. F3A2                 2.8%
            51. F4A1                 2.8%
            52. C3A2                 2.8%
            53. C4A1                 2.8%
            54. O1A2                 2.8%
            55. CHIN2                2.8%
            56. CHIN1                2.8%
            57. O2A1                 2.8%
            58. C3                   1.9%
            59. P3                   1.9%
            60. F8                   1.9%
            61. Fp2                  1.9%
            62. FP1                  1.9%
            63. F3                   1.9%
            64. CHIN                 1.9%
            65. milo                 1.9%
            66. Posizione            1.9%
            67. T6                   1.9%
            68. T5                   1.9%
            69. T3                   1.9%
            70. F7                   1.9%
            71. EOG-R                1.9%
            72. A1                   1.9%
            73. EMG2                 1.9%
            74. O1                   1.9%
            75. O2                   1.9%
            76. EMG1                 1.9%
            77. ECG1                 1.9%
            78. ECG2                 1.9%
            79. A2                   1.9%
            80. C4                   1.9%
            81. EMG-EMG              1.9%
            82. EOG-L                1.9%
            83. F4                   1.9%
            84. P4                   1.9%
            85. Flow                 0.9%
            86. ekg                  0.9%
            87. toracico             0.9%
            88. deltoide             0.9%
            89. Heart Rate Varia     0.9%
            90. EOG dx               0.9%
            91. EOG sin              0.9%
            92. tib dx               0.9%
            93. O1-A2                0.9%
            94. ROC / A1             0.9%
            95. LOC / A2             0.9%
            96. tib sin              0.9%
            97. cannula              0.9%
            98. Flattening           0.9%
            99. DX2                  0.9%
            100. Abdo                 0.9%
            101. Torace               0.9%
            102. Canula               0.9%
            103. EMG                  0.9%
            104. DX1                  0.9%
            105. SX2                  0.9%
            106. SX1                  0.9%
            107. Sound                0.9%
            108. F1-F3                0.9%
            109. Tib dx               0.9%
            110. Tib sx               0.9%
            111. abdomen              0.9%
            112. flow                 0.9%
            113. thorax               0.9%

        Classes:
            - MT
            - R
            - REM
            - S1
            - S2
            - S3
            - S4
            - W
    """

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