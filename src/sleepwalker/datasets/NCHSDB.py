from __future__ import annotations
from datetime import datetime, timedelta

import pandas as pd
from .Basedataset import BaseDataset  


class NCHSDB(BaseDataset):
    """Dataset Summary
        Summary of core statistics for this dataset.

        n_patients: 3960 
        Durations:
            min     : 0 days 00:03:00
            max     : 0 days 16:32:52
            mean    : 0 days 10:15:43.810688002
            median  : 0 days 10:22:16
            q25     : 0 days 09:34:15.500000
            q75     : 0 days 11:07:44

        Channels:
            1. EEG C3-M2            99.7%
            2. EEG O2-M1            99.7%
            3. EEG O1-M2            99.7%
            4. Resp Rate            99.6%
            5. Capno                99.6%
            6. EtCO2                99.6%
            7. Rate                 99.6%
            8. EEG F3-M2            99.6%
            9. Snore                99.6%
            10. EEG C4-M1            99.4%
            11. EEG F4-M1            99.4%
            12. EOG LOC-M2           98.7%
            13. EOG ROC-M1           98.7%
            14. Pressure             70.9%
            15. Resp Abdominal       70.8%
            16. Resp Thoracic        70.8%
            17. Resp Airflow         70.8%
            18. ECG EKG2-EKG         70.8%
            19. EEG CZ-O1            70.8%
            20. SpO2                 70.8%
            21. Tidal Vol            70.7%
            22. Resp PTAF            70.7%
            23. EMG LLeg-RLeg        70.7%
            24. C-flow               70.1%
            25. Patient Event        68.3%
            26. EMG Chin1-Chin2      66.2%
            27. TcCO2                35.6%
            28. EEG Cz-O1            28.9%
            29. C-Flow               28.9%
            30. OSAT                 28.9%
            31. XFlow                28.8%
            32. Resp Abdomen         28.8%
            33. Resp Chest           28.8%
            34. Flow_DR              28.8%
            35. EMG LLEG+-LLEG-      28.8%
            36. ECG LA-RA            28.8%
            37. Snore_DR             28.8%
            38. EMG CHIN1-CHIN2      28.8%
            39. EMG RLEG+-RLEG-      28.8%
            40. C-Pressure           28.8%
            41. Resp Flow            28.8%
            42. EEG Chin1-Chin2      3.4%
            43. EEG LOC-M2           0.9%
            44. EEG ROC-M1           0.8%
            45. EMG Chin3-Chin2      0.5%
            46. EMG Chin2-Chin1      0.5%
            47. EEG O2               0.3%
            48. EEG C3               0.3%
            49. EEG C4               0.3%
            50. EEG O1               0.3%
            51. EEG M1               0.3%
            52. EEG E2               0.3%
            53. EEG E1               0.3%
            54. EEG F4               0.3%
            55. EEG F3               0.3%
            56. EEG 21               0.3%
            57. EEG 22               0.3%
            58. EEG 23               0.3%
            59. EEG 24               0.3%
            60. EEG 25               0.3%
            61. EEG M2               0.3%
            62. EEG Chin1            0.3%
            63. EEG Chin2            0.3%
            64. EEG Chin3            0.3%
            65. EEG EKG1             0.3%
            66. EEG EKG2             0.3%
            67. EEG RLeg1            0.3%
            68. EEG RLeg2            0.3%
            69. EEG LLeg1            0.3%
            70. EEG LLeg2            0.3%
            71. EEG 20               0.3%
            72. EEG 29               0.3%
            73. EEG 30               0.3%
            74. EEG 31               0.3%
            75. EEG 32               0.3%
            76. EEG 33               0.3%
            77. EEG 26               0.3%
            78. EEG 27               0.3%
            79. EEG 28               0.3%
            80. EEG Therm            0.3%
            81. EEG Spare            0.3%
            82. EEG Snore            0.3%
            83. EEG Press            0.3%
            84. EEG 40               0.3%
            85. EEG Abd              0.3%
            86. EEG Chest            0.3%
            87. EEG ROC-M2           0.2%
            88. EEG C4-M2            0.2%
            89. EEG F4-M2            0.2%
            90. PTAF                 0.2%
            91. EMG CHIN1-CHIN3      0.2%
            92. EEG Chin3-Chin2      0.1%
            93. EEG EKG2-EKG         0.1%
            94. EMG LLEG-RLEG        0.1%
            95. ECG ECGL-ECGR        0.1%
            96. EMG LAT1-LAT2        0.1%
            97. EMG RAT1-RAT2        0.1%
            98. SNORE_DR             0.1%
            99. Resp FLOW-Ref        0.1%
            100. EEG EKG-RLeg         0.1%
            101. EEG Chin1-Chin3      0.0%
            102. EMG RLEG-RLEG2       0.0%
            103. EMG Chin1-Chin3      0.0%
            104. Resp Airflow+-Re     0.0%
            105. Position             0.0%
            106. Chin1                0.0%
            107. Chin2                0.0%
            108. Fp1                  0.0%
            109. Fp2                  0.0%
            110. F7                   0.0%
            111. F8                   0.0%
            112. F3                   0.0%
            113. F4                   0.0%
            114. T3                   0.0%
            115. T4                   0.0%
            116. C3                   0.0%
            117. C4                   0.0%
            118. T5                   0.0%
            119. EMG LLEG-LLEG2       0.0%
            120. T6                   0.0%
            121. P3                   0.0%
            122. O1                   0.0%
            123. P4                   0.0%
            124. FZ                   0.0%
            125. CZ                   0.0%
            126. PZ                   0.0%
            127. O2                   0.0%
            128. M2                   0.0%
            129. RLeg                 0.0%
            130. LLeg                 0.0%
            131. ROC                  0.0%
            132. LOC                  0.0%
            133. FPZ                  0.0%
            134. OZ                   0.0%
            135. M1                   0.0%
            136. Chin3                0.0%
            137. EKG2                 0.0%
            138. Airflow              0.0%
            139. EKG                  0.0%
            140. Abdominal            0.0%
            141. 38                   0.0%
            142. 39                   0.0%
            143. Thoracic             0.0%
            144. 40                   0.0%
            145. DC8                  0.0%
            146. DC4                  0.0%
            147. DC3                  0.0%
            148. PPG                  0.0%
            149. Pleth                0.0%
            150. OSat                 0.0%
            151. PR                   0.0%

        Classes:
            - Sleep stage 1
            - Sleep stage 2
            - Sleep stage 3
            - Sleep stage ?
            - Sleep stage N1
            - Sleep stage N2
            - Sleep stage N3
            - Sleep stage R
            - Sleep stage W
    """


    def __init__(self, label_should_contain = "Sleep stage", **kwargs):
        self.label_should_contain = label_should_contain
        super().__init__(**kwargs)

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        """
        Load TSV annotations for the NCHSDB dataset.

        Args:
            edf_path: Path to the EDF file (used to locate .tsv annotation file)
            start_datetime: Timestamp indicating when the EDF recording started

        Returns:
            pd.DataFrame with columns:
                ['Label', 'Starttime', 'Endtime', 'Duration']
        """
        annot_path = f"{edf_path.split('.edf')[0]}.tsv"
        df = pd.read_csv(annot_path, sep="\t")

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Compute proper datetime columns
        # onset and duration are in seconds from start_datetime
        start_times = []
        end_times = []

        for onset, dur in zip(df["onset"], df["duration"]):
            start_ts = start_datetime + pd.to_timedelta(onset, unit="s")
            end_ts = start_ts + pd.to_timedelta(dur, unit="s")

            start_times.append(start_ts)
            end_times.append(end_ts)
        
        df["Label"] = df["description"].astype(str).str.strip()
        df["Starttime"] = start_times
        df["Endtime"] = end_times
        df["Duration"] = pd.to_timedelta(df["duration"], unit="s")

        if self.label_should_contain is not None and self.label_should_contain != "":
            df.loc[~df["Label"].str.contains(self.label_should_contain, na=False), :] = None
            df = df.dropna()

        return df[["Label", "Starttime", "Endtime", "Duration"]] 