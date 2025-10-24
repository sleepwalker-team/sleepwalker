from __future__ import annotations
from datetime import datetime, timedelta

import os
from pathlib import Path
import re
import pandas as pd
from .Basedataset import BaseDataset  

class WSC(BaseDataset):
    """
    Dataset Summary
    Summary of core statistics for this dataset.
    n_patients: 49 
    Durations:
        min     : 0 days 00:19:30
        max     : 0 days 10:30:00
        mean    : 0 days 03:59:58.034830893
        median  : 0 days 02:21:00
        q25     : 0 days 01:55:30
        q75     : 0 days 07:32:08
    Channels:
        1. ECG                  97.1%
        2. R OCC                65.1%
        3. L OCC                65.1%
        4. CHIN EMG             65.1%
        5. R CENT               65.1%
        6. L CENT               65.1%
        7. L EOG                65.1%
        8. R EOG                65.1%
        9. spo2                 32.1%
        10. abdomen              32.1%
        11. sum                  32.1%
        12. lleg_r               32.1%
        13. E1                   32.1%
        14. E2                   32.1%
        15. thorax               32.1%
        16. position             32.1%
        17. nas_pres             32.1%
        18. snore                32.1%
        19. C3_M2                32.0%
        20. O1_M2                32.0%
        21. chin                 28.6%
        22. nasalflow            28.2%
        23. oralflow             28.2%
        24. flow                 3.8%
        25. F3_M2                3.8%
        26. cchin_l              3.4%
        27. EKG1-EKG2            2.9%
        28. O1-M2                2.6%
        29. Chin1-Chin2          2.6%
        30. REOG-M1              2.6%
        31. C3-M2                2.6%
        32. LEOG-M2              2.6%
        33. O2-M1                2.6%
        34. C4-M1                2.6%
        35. Pz_M2                0.3%
        36. Fz_M2                0.3%
        37. Cz_M2                0.3%
        38. LEOG-AVG             0.3%
        39. REOG-AVG             0.3%
        40. C3-AVG               0.3%
        41. C4-AVG               0.3%
        42. O2-AVG               0.3%
        43. O1-AVG               0.3%
        44. Chin1-Chin3          0.2%
        45. Chin3-Chin2          0.1%
        46. Pz_Cz                0.1%
        47. F3_M1                0.1%
        48. C3_M1                0.1%
        49. O1_M1                0.1%
        50. O2_M1                0.1%
        51. cchin_r              0.1%
        52. Fz_M1                0.1%
        53. Cz_M1                0.1%
    Classes:
        - n1
        - n2
        - n3
        - no stage
        - r
        - w
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        folder = Path(edf_path).parent
        name = Path(edf_path).name.split(".edf")[0]
        annot_path = os.path.join(folder, f"{name}.allscore.txt")
        
        # Read file as two columns: time + event description
        df = pd.read_csv(annot_path, sep="\t", engine="python", names=["time", "event"], comment="#")
        df["time"] = df["time"].str.strip()
        df["event"] = df["event"].astype(str).str.strip()

        # Prepare base date (00:00 same day as EDF start)
        base_date = pd.Timestamp(start_datetime).normalize()
        start_times, end_times, labels = [], [], []

        prev_ts = None
        for t_str, ev in zip(df["time"], df["event"]):
            # Parse clock time (e.g., "21:33:01.00")
            try:
                t_obj = pd.to_datetime(t_str, format="%H:%M:%S.%f")
            except ValueError:
                t_obj = pd.to_datetime(t_str, format="%H:%M:%S", errors="coerce")
            ts = pd.Timestamp.combine(base_date.to_pydatetime().date(), t_obj.time())

            # Handle midnight rollover
            if prev_ts is not None and ts < prev_ts:
                ts += pd.Timedelta(days=1)
                base_date += pd.Timedelta(days=1)
            prev_ts = ts

            # --- Case 1: STAGE event ---
            if "stage" in ev.lower():
                label = ev.split("STAGE -")[-1].strip().lower()
                duration = pd.Timedelta(seconds=30)
                start_times.append(ts)
                end_times.append(ts + duration)
                labels.append(label)
                continue

            # --- Case 2: has duration (DUR: X SEC.) ---
            dur_match = re.search(r"DUR:\s*([0-9.]+)\s*SEC", ev, re.IGNORECASE)
            if dur_match:
                duration = pd.to_timedelta(float(dur_match.group(1)), unit="s")
                label = re.split(r"DUR:.*?SEC\.?-", ev, flags=re.IGNORECASE)[-1].strip().lower()
                start_times.append(ts)
                end_times.append(ts + duration)
                labels.append(label)
                continue

            # --- Drop everything else ---
            continue

        df = pd.DataFrame({
            "Label": labels,
            "Starttime": start_times,
            "Endtime": end_times,
            "Duration": [e - s for s, e in zip(start_times, end_times)]
        })

        return df[["Label", "Starttime", "Endtime", "Duration"]]