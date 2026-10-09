"""SHHS dataset adapter for NSRR and Profusion XML sidecars."""

import os

import pandas as pd
from sleepwalker.datasets.Basedataset import BaseDataset
from sleepwalker.datasets.utils import read_nsrr, read_profusion

def correct_shhs_waveform(values, *, unit, is_recording):
    """Use relative units for SHHS respiratory inputs with missing dimensions.

    SHHS equipment documentation distinguishes older respiratory inputs from
    newer airflow on an auxiliary input with 250 microvolt sensitivity:
    https://sleepdata.org/datasets/shhs/pages/08-equipment-shhs1.md
    https://sleepdata.org/forum/shhs-database-new-air-signal
    Local October 2026 inspection found blank-unit AIRFLOW, ABDO RES and THOR
    RES with dummy +/-1 physical ranges. EDF FAQ Q8 permits such uncalibrated
    thermocouple exports: https://www.edfplus.info/specs/edffaq.html
    This recipe declares their scale relative; it does not infer a voltage gain
    or divide by 250. Already labelled signals and digital counts are retained.
    Apply this function only to the respiratory channels named above.
    """
    return values, "relative" if unit is None else unit


def correct_shhs_saturation(values, *, unit, is_recording):
    """Declare missing SHHS SaO2/SpO2 dimensions as percent.

    Local October 2026 inspection found blank dimensions with a 0..100 physical
    mapping. SHHS describes pulse oximetry here:
    https://sleepdata.org/datasets/shhs/pages/08-equipment-shhs1.md
    This is an explicit dataset recipe assumption. It preserves decoded values,
    gain, offset, labelled variants and digital counts. It does not check ranges.
    """
    return values, "%" if unit is None else unit


def get_preprocessors(channel_name):
    """Return explicit unit-label corrections for one SHHS source channel."""
    if channel_name.upper() in {"AIRFLOW", "ABDO RES", "THOR RES"}:
        return [correct_shhs_waveform]
    if channel_name.casefold() in {"sao2", "spo2"}:
        return [correct_shhs_saturation]
    return []


def read_xml(fpath, annotator, start_date):
    """Load one SHHS XML annotation file into the common event-table format."""
    name = os.path.basename(fpath).split(".edf")[0]
    subset = name.split("-")[0]

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

class SHHS(BaseDataset):
    """Read SHHS EDF files together with NSRR or Profusion XML annotations."""

    def __init__(self, 
            annotator = "nsrr",
            ignore_patients_with_partial_events = False,
            **kwargs
        ): 
        """Configure the SHHS adapter.

        Args:
            annotator: Primary annotation source to read.
            ignore_patients_with_partial_events: Whether to reject patients that
                do not cover each mapped label at least once.
            **kwargs: Forwarded to :class:`BaseDataset`.
        """
        if annotator not in ["nsrr", "profusion"]:
            raise ValueError(f"Unknown value for annotator given. Received {annotator}, but expected {{nsrr, profusion}}")
        else:
            self.annotator = annotator

        self.ignore_patients_with_partial_events = ignore_patients_with_partial_events
        super().__init__(**kwargs)

    def has_extra_target(self):
        """Return whether the alternate annotation source is available."""
        return True

    def get_extra_event_df(self, fpath, start_date):
        """Load the alternate annotator XML as ``target_extra``."""
        if self.annotator == "profusion":
            df = read_xml(fpath, "nsrr", start_date)
        else:
            df = read_xml(fpath, "profusion", start_date)

        return df 

    def get_event_df(self, fpath, start_date):
        """Load the primary SHHS annotation source for one EDF file."""
        df = read_xml(fpath, self.annotator, start_date)

        if self.ignore_patients_with_partial_events and len(self.event_mapping) > 0:
            vals = sorted(set([v for k, v in self.event_mapping.items() if k != "default_class"]))
            labels = sorted(set(df["Label"].unique()))
            if labels != vals:
                raise ValueError(f"Patient did not contain at-least one window with one of the events attached. Patient has {labels}, but expected {vals}.")        
        return df
