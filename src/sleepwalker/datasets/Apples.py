"""APPLES dataset adapter.

This adapter reads APPLES EDF files together with tab-delimited ``.annot``
sidecars. It is used by the repository's sleep-staging workflow and exposes the
annotation labels largely as stored in the source files.
"""

from __future__ import annotations
from datetime import datetime, timedelta

import pandas as pd
from .Basedataset import BaseDataset  

def correct_apples_eeg(values, *, unit, is_recording):
    """Apply NSRR's microvolt assumption to missing APPLES EEG/EOG units.

    NSRR's reproducible APPLES documentation, sections 1, 7 and 10, describes
    Alice4 acquisition and this assumption for missing EEG/EOG dimensions:
    https://gitlab-scm.partners.org/zzz-public/nsrr/-/blob/master/studies/apples/README.md
    Keep the EDF physical amplitude mapping. This only supplies a unit label;
    already labelled inputs and digital counts retain their units.
    """
    return values, "uV" if unit is None else unit


def correct_apples_waveform(values, *, unit, is_recording):
    """Declare missing APPLES ECG/EMG/respiratory dimensions as relative.

    NSRR reports ECG/EMG/LEG offsets around 100 and variable respiratory/EMG
    scales between and within sites; it establishes no universal voltage gain:
    https://gitlab-scm.partners.org/zzz-public/nsrr/-/blob/master/studies/apples/README.md
    Do not infer calibration from a 0..255 range. Preserve physical decoding,
    labelled variants and digital counts. Later processors may center or scale
    these values explicitly, but cannot claim a recovered voltage calibration.
    """
    return values, "relative" if unit is None else unit


def correct_apples_saturation(values, *, unit, is_recording):
    """Declare missing APPLES SpO2/SaO2 dimensions as percent.

    Local October 2026 audit: 20 files have '.' dimensions. Sixteen have 0..255
    identity mappings; four have non-identity ranges near -13..111/115 with
    samples consistent with percent. This explicit recipe assumes percent,
    without replacing either EDF gain/offset mapping or inferring sensor volts.
    Acquisition context: NSRR's APPLES Alice4 processing documentation:
    https://gitlab-scm.partners.org/zzz-public/nsrr/-/blob/master/studies/apples/README.md
    Preserve labelled variants and digital counts. This is a unit correction,
    not a signal-quality or physiological range check.
    """
    return values, "%" if unit is None else unit


def get_preprocessors(channel_name):
    """Return explicit unit-label corrections for one APPLES source channel."""
    name = channel_name.casefold()
    if name in {"c3_m2", "c4_m1", "o1_m2", "o2_m1", "c3-m2", "c4-m1", "o1-m2", "o2-m1", "loc", "roc"}:
        return [correct_apples_eeg]
    if name in {"emg", "ecg", "leg", "abdomen", "thorax", "thermistor", "nasal_pres", "flow_patient", "flow_patient.1"}:
        return [correct_apples_waveform]
    if name in {"spo2", "sao2"}:
        return [correct_apples_saturation]
    return []


def convert_to_datetime(row, start_date):
    """Convert APPLES time strings into absolute datetimes."""
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
    """
    Dataset URL: 
        https://sleepdata.org/datasets/apples/pages/README.md
        https://gitlab-scm.partners.org/zzz-public/nsrr/-/tree/master/studies/apples

    Dataset Summary
        Summary of core statistics for this dataset.

        n_patients: 1084 
        Durations:
            min     : 0 days 00:03:12
            max     : 0 days 11:15:00
            mean    : 0 days 08:10:24.836956522
            median  : 0 days 08:13:30
            q25     : 0 days 07:56:30
            q75     : 0 days 08:40:00

        Channels:
            1. snore                100.0%
            2. SpO2                 100.0%
            3. LOC                  100.0%
            4. EMG                  100.0%
            5. ECG                  99.9%
            6. O2_M1                99.4%
            7. C4_M1                99.4%
            8. C3_M2                99.4%
            9. thorax               99.4%
            10. thermistor           99.4%
            11. abdomen              99.4%
            12. nasal_pres           99.4%
            13. ROC                  99.4%
            14. O1_M2                99.2%
            15. LEG                  79.8%
            16. pulse                70.1%

        Classes:
            - L
            - LM
            - N1
            - N2
            - N3
            - R
            - W
            - apnea
            - arousal
            - desat
            - epos2_mixed
            - epos2_nonsupine
            - epos2_supine
            - epos5_left
            - epos5_mixed
            - epos5_prone
            - epos5_right
            - epos5_supine
            - epos5_upright
            - epos9_left
            - epos9_mixed
            - epos9_prone
            - epos9_prone_left
            - epos9_prone_right
            - epos9_right
            - epos9_supine
            - epos9_supine_left
            - epos9_supine_right
            - epos9_upright
            - hypopnea
            - pos2_nonsupine
            - pos2_supine
            - pos5_left
            - pos5_prone
            - pos5_right
            - pos5_supine
            - pos5_upright
            - pos9_left
            - pos9_prone
            - pos9_prone_left
            - pos9_prone_right
            - pos9_right
            - pos9_supine
            - pos9_supine_left
            - pos9_supine_right
            - pos9_upright
            - snoring
        """
    def __init__(self, **kwargs):
        """Initialize the APPLES adapter.

        Args:
            **kwargs: Forwarded to :class:`BaseDataset`.
        """
        super().__init__(**kwargs)

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        """Load one APPLES ``.annot`` sidecar into event-table form.

        Args:
            edf_path: Path to the EDF recording.
            start_datetime: Recording start timestamp from the EDF header.

        Returns:
            A dataframe with labels, absolute start and end times, and a
            computed duration column.
        """
        annot_path = f"{edf_path.split('.edf')[0]}.annot"

        df = pd.read_csv(annot_path, sep="\t", dtype={"class":str, "instance":str, "channel":str, "meta":str})
        df = df.rename(columns={"class":"Label", "start":"Starttime", "stop":"Endtime"})

        df['Starttime'] = df['Starttime'].apply(lambda row: convert_to_datetime(row, start_datetime))
        df['Endtime'] = df['Endtime'].apply(lambda row: convert_to_datetime(row, start_datetime))

        df["Duration"] = df["Endtime"] - df["Starttime"]

        return df[["Label", "Starttime", "Endtime", "Duration"]]
