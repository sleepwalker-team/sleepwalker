"""ABC dataset adapter for NSRR and Profusion event sidecars.

The module reads EDF recordings together with XML annotations stored in parallel
``annotations-events-*`` folders. Current training scripts use it in the
sleep-staging workflow and may also compare NSRR and Profusion variants through
the adapter's extra-target support.
"""

import os

import pandas as pd
from sleepwalker.datasets.Basedataset import BaseDataset
from sleepwalker.utils import logger
from sleepwalker.datasets.utils import read_nsrr, read_profusion
from pathlib import Path

def read_xml(fpath, annotator, start_date):
    """Load one ABC XML annotation file into the common event-table format."""
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

class ABC(BaseDataset):
    """
    Dataset URL: https://sleepdata.org/datasets/abc/pages/montage-and-sampling-rate-information.md

    Dataset Summary
        Summary of core statistics for this dataset.

        n_patients: 132 
        Durations:
            min     : 0 days 07:15:19
            max     : 0 days 09:53:13
            mean    : 0 days 08:24:25.356060606
            median  : 0 days 08:20:02.500000
            q25     : 0 days 08:12:35.250000
            q75     : 0 days 08:32:27.750000

        Channels:
            1. F3                   100.0%
            2. F4                   100.0%
            3. C3                   100.0%
            4. C4                   100.0%
            5. O1                   100.0%
            6. O2                   100.0%
            7. M1                   100.0%
            8. M2                   100.0%
            9. E1                   100.0%
            10. E2                   100.0%
            11. ECG1                 100.0%
            12. ECG2                 100.0%
            13. LLeg1                100.0%
            14. LLeg2                100.0%
            15. RLeg1                100.0%
            16. RLeg2                100.0%
            17. Chin1                100.0%
            18. Chin2                100.0%
            19. Chin3                100.0%
            20. Airflow              100.0%
            21. Abdo                 100.0%
            22. Thor                 100.0%
            23. Snore                100.0%
            24. Sum                  100.0%
            25. PosSensor            100.0%
            26. Ox Status            100.0%
            27. Pulse                100.0%
            28. SpO2                 100.0%
            29. Nasal Pressure       100.0%
            30. CPAP Flow            100.0%
            31. CPAP Press           100.0%
            32. Pleth                100.0%
            33. Derived HR           100.0%
            34. Light                100.0%
            35. Manual Pos           100.0%
            36. Respiratory Rate     29.5%

        Classes:
            - arousal resulting from respiratory effort|arousal (aro res)
            - arousal|arousal ()
            - blood pressure artifact|blood pressure artifact
            - body temperature artifact|body temperature artifact
            - central apnea|central apnea
            - distal ph|distal ph
            - etco2 artifact|etco2 artifact
            - hypopnea|hypopnea
            - limb movement - left|limb movement (left)
            - limb movement - right|limb movement (right)
            - obstructive apnea|obstructive apnea
            - periodic leg movement - left|plm (left)
            - periodic leg movement - right|plm (right)
            - proximal ph artifact|proximal ph artifact
            - proximal ph|distal ph artifact
            - rem sleep|5
            - respiratory artifact|respiratory artifact
            - spo2 artifact|spo2 artifact
            - spo2 desaturation|spo2 desaturation
            - stage 1 sleep|1
            - stage 2 sleep|2
            - stage 3 sleep|3
            - technician notes
            - unscored|9
            - unsure|unsure
            - wake|0
        """

    def __init__(self, 
            annotator = "nsrr",
            ignore_patients_with_partial_events = False,
            **kwargs
        ): 
        """Configure the ABC dataset adapter.

        Args:
            annotator: Which XML annotation variant to expose as the primary
                target.
            ignore_patients_with_partial_events: Whether to reject patients that
                do not cover every mapped label at least once.
            **kwargs: Forwarded to :class:`BaseDataset`.
        """
        if annotator not in ["nsrr", "profusion"]:
            raise ValueError(f"Unknown value for annotator given. Received {annotator}, but expected {{nsrr, profusion}}")
        else:
            self.annotator = annotator

        self.ignore_patients_with_partial_events = ignore_patients_with_partial_events
        super().__init__(**kwargs)

    def has_extra_target(self):
        """Return whether the alternate annotator is exposed as extra target."""
        return True

    def get_extra_event_df(self, fpath, start_date):
        """Load the alternate annotation source as ``target_extra``."""
        if self.annotator == "profusion":
            df = read_xml(fpath, "nsrr", start_date)
        else:
            df = read_xml(fpath, "profusion", start_date)

        return df 

    def get_event_df(self, fpath, start_date):
        """Load the primary annotation source for one ABC recording."""
        df = read_xml(fpath, self.annotator, start_date)

        if self.ignore_patients_with_partial_events and len(self.event_mapping) > 0:
            vals = sorted(set([v for k, v in self.event_mapping.items() if k != "default_class"]))
            labels = sorted(set(df["Label"].unique()))
            if labels != vals:
                raise ValueError(f"Patient did not contain at-least one window with one of the events attached. Patient has {labels}, but expected {vals}.")        
        return df
