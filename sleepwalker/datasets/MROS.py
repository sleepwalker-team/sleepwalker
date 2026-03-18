import os

import pandas as pd
from sleepwalker.datasets.Basedataset import BaseDataset
from sleepwalker.datasets.utils import read_nsrr, read_profusion
from pathlib import Path

def read_xml(fpath, annotator, start_date):
    name = Path(fpath).name.split(".edf")[0]
    subset = Path(fpath).parent.name
    repo_folder = fpath.split("edfs")[0]
    
    xml_path = os.path.join(repo_folder, f"annotations-events-{annotator}", subset, f"{name}-{annotator}.xml")
    if annotator == "nsrr":
        xml_df = read_nsrr(xml_path)
    else:
        xml_df = read_profusion(xml_path)

    xml_df["Starttime"] = xml_df.apply(lambda row : start_date + pd.to_timedelta(f"{row['Start']} s"),axis=1)
    xml_df["Endtime"] = xml_df.apply(lambda row : start_date + pd.to_timedelta(f"{row['Start']} s" + pd.to_timedelta(f"{row.Duration}s")),axis=1)

    xml_df = xml_df[["Label", "Starttime", "Endtime"]].dropna()

    if len(xml_df) == 0:
        raise ValueError(f"Cannot read XML file for {xml_path}. File is empty?")
    
    return xml_df

class MROS(BaseDataset):
    """
    Dataset URL:
        https://sleepdata.org/datasets/mros
        https://sleepdata.org/datasets/mros/pages/equipment-mros1.md

    Summary of core statistics for this dataset.
        n_patients: 3839 
        Durations:
            min     : 0 days 04:00:00
            max     : 1 days 10:59:03
            mean    : 0 days 11:29:12.651699974
            median  : 0 days 11:10:00
            q25     : 0 days 10:02:30
            q75     : 0 days 12:24:00

        Channels:
            1. SUM                  100.0%
            2. Airflow              100.0%
            3. HR                   99.9%
            4. STAT                 99.9%
            5. Position             99.9%
            6. C3                   99.8%
            7. C4                   99.8%
            8. DHR                  99.8%
            9. Leg L                73.4%
            10. ROC                  73.4%
            11. Thoracic             73.4%
            12. LOC                  73.4%
            13. Abdominal            73.4%
            14. SaO2                 73.4%
            15. Leg R                73.4%
            16. ECG R                73.3%
            17. L Chin               73.3%
            18. ECG L                73.3%
            19. R Chin               73.3%
            20. A1                   73.2%
            21. A2                   73.2%
            22. Cannula Flow         72.6%
            23. M1                   26.6%
            24. M2                   26.6%
            25. LegL                 26.6%
            26. LegR                 26.6%
            27. ECGL                 26.6%
            28. E1                   26.6%
            29. E2                   26.6%
            30. ECGR                 26.6%
            31. RChin                26.6%
            32. LChin                26.6%
            33. Chest                26.6%
            34. ABD                  26.6%
            35. SpO2                 26.6%
            36. CannulaFlow          25.4%
            37. L Chin-R Chin        0.2%
            38. C3-A2                0.2%
            39. C4-A1                0.2%
            40. ECG L-ECG R          0.2%
            41. CH37                 0.1%
            42. CH36                 0.0%

        Classes:
            - arousal resulting from respiratory effort|arousal (aro res)
            - arousal|arousal ()
            - arousal|arousal (aro limb)
            - arousal|arousal (asda)
            - asda arousal|arousal (asda)
            - central apnea|central apnea
            - hypopnea|hypopnea
            - limb movement - left|limb movement (left)
            - limb movement - right|limb movement (right)
            - mixed apnea|mixed apnea
            - narrow complex tachycardia|narrow complex tachycardia
            - obstructive apnea|obstructive apnea
            - periodic breathing|periodic breathing
            - periodic leg movement - left|plm (left)
            - periodic leg movement - right|plm (right)
            - rem sleep|5
            - respiratory artifact|respiratory artifact
            - respiratory effort related arousal|rera
            - spo2 artifact|spo2 artifact
            - spo2 desaturation|spo2 desaturation
            - spontaneous arousal|arousal (aro spont)
            - stage 1 sleep|1
            - stage 2 sleep|2
            - stage 3 sleep|3
            - stage 4 sleep|4
            - tachycardia|tachycardia
            - unscored|9
            - unsure|unsure
            - wake|0
    """
    def __init__(self, 
            annotator = "nsrr",
            ignore_patients_with_partial_events = False,
            **kwargs
        ): 
        
        if annotator not in ["nsrr", "profusion"]:
            raise ValueError(f"Unknown value for annotator given. Received {annotator}, but expected {{nsrr, profusion}}")
        else:
            self.annotator = annotator

        self.ignore_patients_with_partial_events = ignore_patients_with_partial_events
        super().__init__(**kwargs)

    def has_extra_target(self):
        return True

    def get_extra_event_df(self, fpath, start_date):
        if self.annotator == "profusion":
            df = read_xml(fpath, "nsrr", start_date)
        else:
            df = read_xml(fpath, "profusion", start_date)

        return df 

    def get_event_df(self, fpath, start_date):
        df = read_xml(fpath, self.annotator, start_date)

        if self.ignore_patients_with_partial_events and len(self.event_mapping) > 0:
            vals = sorted(set([v for k, v in self.event_mapping.items() if k != "default_class"]))
            labels = sorted(set(df["Label"].unique()))
            if labels != vals:
                raise ValueError(f"Patient did not contain at-least one window with one of the events attached. Patient has {labels}, but expected {vals}.")        
        return df
