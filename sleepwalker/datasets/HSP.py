from __future__ import annotations
from datetime import datetime, timedelta

import pandas as pd
from .Basedataset import BaseDataset  
from os.path import basename, dirname, join, exists

class HSP(BaseDataset):
    '''
        Dataset Summary
            Summary of core statistics for this dataset.

            Durations:
                min     : 0 days 00:00:01
                max     : 0 days 22:30:40
                mean    : 0 days 07:31:51
                median  : 0 days 07:40:56
                q25     : 0 days 07:09:05
                q75     : 0 days 08:09:35

            Channels (473 total):
                1. ABD                       97.8%
                2. CHEST                     97.5%
                3. E1-M2                     97.4%
                4. C3-M2                     96.2%
                5. O1-M2                     96.2%
                6. C4-M1                     95.8%
                7. F3-M2                     95.8%
                8. O2-M1                     95.7%
                9. F4-M1                     95.3%
                10. PTAF                      92.9%
                11. IC                        82.9%
                12. EKG                       82.0%
                13. LAT                       81.4%
                14. RAT                       81.0%
                15. SNORE                     80.1%
                16. SaO2                      78.7%
                17. HR                        78.7%
                18. AIRFLOW                   77.5%
                19. CHIN1-CHIN2               60.3%
                20. CFLOW                     56.9%
                21. E2-M1                     56.4%
                22. Leak                      51.8%
                23. E2-M2                     41.1%
                24. Chin1-Chin2               34.5%
                25. CPRES                     33.5%
                26. EtCO2                     27.7%
                27. DC8                       21.5%
                28. Pleth                     21.3%
                29. DC10                      21.3%
                30. DC11                      21.3%
                31. DC9                       21.3%
                32. PR                        21.3%
                33. M2                        21.3%
                34. P3                        21.3%
                35. P4                        21.3%
                36. M1                        21.3%
                37. F7                        21.3%
                38. F8                        21.3%
                39. Fp1                       21.2%
                40. Fp2                       21.2%
                41. DC7                       20.8%
                42. CO2 Wave                  20.5%
                43. AirFlow                   20.5%
                44. SpO2                      19.8%
                45. Snore                     19.8%
                46. LEAK                      19.6%
                47. CPAP                      19.5%
                48. CFlow                     19.1%
                49. DC14                      18.8%
                50. DC1                       18.1%
                51. Cz                        18.1%
                52. PPG                       18.0%
                53. XVolume                   18.0%
                54. Elevation                 18.0%
                55. Activity                  18.0%
                56. T7                        18.0%
                57. DIF1+                     18.0%
                58. P7                        18.0%
                59. P8                        18.0%
                60. DIF5+                     18.0%
                61. X32                       18.0%
                62. Pz                        18.0%
                63. T8                        18.0%
                64. DIF4-                     18.0%
                65. DIF1-                     18.0%
                66. DIF2+                     18.0%
                67. DIF2-                     18.0%
                68. DIF3-                     18.0%
                69. DIF3+                     18.0%
                70. DIF4+                     18.0%
                71. Position                  18.0%
                72. DIF8-                     18.0%
                73. DC16                      18.0%
                74. DIF7+                     18.0%
                75. DC15                      18.0%
                76. RR                        18.0%
                77. RMI                       18.0%
                78. Pressure                  18.0%
                79. Phase                     18.0%
                80. DIF6-                     18.0%
                81. Fpz                       18.0%
                82. PTT                       18.0%
                83. Fz                        18.0%
                84. Oz                        18.0%
                85. ECG-V2                    18.0%
                86. TRIG                      18.0%
                87. ECG-V1                    18.0%
                88. DIF5-                     18.0%
                89. DIF8+                     18.0%
                90. XFlow                     18.0%
                91. DIF7-                     18.0%
                92. XSum                      18.0%
                93. DIF6+                     18.0%
                94. DC12                      18.0%
                95. ECG-LL                    18.0%
                96. PulseQuality              18.0%
                97. DIF10+                    17.8%
                98. DIF9+                     17.8%
                99. DIF10-                    17.8%
                100. DIF9-                     17.8%
                101. Eye Down                  17.3%
                102. Eye Up                    17.3%
                103. CHIN3                     17.2%
                104. Arm1                      17.2%
                105. Arm2                      17.2%
                106. Snore_DR                  17.0%
                107. IPAP                      17.0%
                108. CHIN2                     16.5%
                109. RLEG+                     16.5%
                110. LLEG-                     16.5%
                111. RLEG-                     16.5%
                112. LLEG+                     16.5%
                113. ECG-RA                    16.5%
                114. ECG-LA                    16.5%
                115. EPAP                      16.5%
                116. IC2                       15.7%
                117. IC1                       15.7%
                118. Airflow2                  15.7%
                119. Tidal                     13.8%
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_event_df(self, edf_path, start_datetime):
        bn = basename(edf_path)
        annot_name = f'{bn.replace('eeg', 'annotations').replace('.edf', '.csv')}'
        annot_path = join(dirname(edf_path), annot_name)

        if not exists(annot_path):
            # Try Xltek annotations
            annot_name = f'{bn.replace('-psg_eeg.edf', '_Xltek.csv')}'
            annot_path = join(dirname(edf_path), annot_name)
            if not exists(annot_path):
                raise RuntimeError('No annotation file found for EDF file', bn)

        df = pd.read_csv(annot_path)
        df = df.rename(columns={'event': 'Label', 'time': 'Starttime', 'duration': 'Duration'})

        base_date = f'{start_datetime._date_repr}'
        df['Starttime'] = pd.to_datetime(base_date + ' ' + df['Starttime'])

        # Any time earlier than the first timestamp has rolled over to the next day
        # Add 1 day wherever the time is less than the starting time
        rollover_mask = df['Starttime'] < df['Starttime'].iloc[0]
        df.loc[rollover_mask, 'Starttime'] += pd.Timedelta(days=1)

        df['Endtime'] = df['Starttime'] + pd.to_timedelta(df['Duration'], unit='s')

        return df[['Label', 'Starttime', 'Endtime', 'Duration']]

