from __future__ import annotations

from abc import ABC, abstractmethod
import bisect
from collections import defaultdict
import copy
from dataclasses import dataclass
from functools import partial
import os
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, cast

import numpy as np
import pandas as pd
import pyedflib
import torch
from torch.utils.data import Dataset

from sleepwalker.utils import logger
from sleepwalker.core.signal import edf_to_df, read_edf_meta
from sleepwalker.datasets.normalizer import Normalizer
import multiprocessing

@dataclass
class ChannelConfig:
    name: str
    normalizer: Optional[Normalizer] = None  

@dataclass
class EDFFile: 
    channels: List[str]
    path: str
    start_date: pd.Timestamp
    handle: Optional[pyedflib.EdfReader] = None
    length: int = 0
    X: Optional[pd.DataFrame] = None
    classes: Optional[Set[str]] = None
    labels: Optional[EventIndex] = None
    labels_extra: Optional[EventIndex] = None
    normalizers: Optional[dict[str, Normalizer]] = None

    def get_x(self, start_date:pd.Timestamp, end_date:pd.Timestamp, sample_frequency, resample_type):
        if self.X is None:
            x_df = edf_to_df(self.path, self.channels, start_date, end_date, sample_frequency, resample_type)
            
            if self.normalizers:
                for col, norm in self.normalizers.items():
                    if col in x_df.columns:
                        vals = x_df[col].to_numpy(dtype=float).reshape(-1, 1)
                        x_df[col] = norm.transform(vals).ravel()

            return x_df
        else:
            return self.X.loc[start_date:end_date]
    
    def get_y_extra(self, start_date: pd.Timestamp, end_date: pd.Timestamp, sample_frequency, classes):
        if self.labels_extra:
            freq = pd.to_timedelta(1.0 / sample_frequency, unit="s")
            return self.labels_extra.query(start_date, end_date, freq=freq, sparse=False, labels=classes)
        else:
            return None

    def get_y(self, start_date: pd.Timestamp, end_date: pd.Timestamp, sample_frequency, classes):
        if self.labels:
            freq = pd.to_timedelta(1.0 / sample_frequency, unit="s")
            return self.labels.query(start_date, end_date, freq=freq, sparse=False, labels=classes)
        else:
            return None

def batch_collate(batch, ignore_list = ["time", "patient"]):
    final_dict = defaultdict(list)

    for b in batch:
        for k, v in b.items():
            final_dict[k].append(v)
    
    return {k: torch.stack(v) if k not in ignore_list else v for k, v in final_dict.items()}

class EventIndex:
    """
    Fast indicator sampling for labeled time intervals.
    Semantics: event is active at t iff start <= t < end  (left-closed, right-open).
    All timestamps should be tz-aligned (ideally UTC).
    """
    def __init__(self, events: pd.DataFrame): #merge_overlaps: bool = True
        # expected columns: starttime, endtime, label
        df = events.copy()

        # self.labels: list = labels #sorted(df["Label"].unique())
        self.starts: Dict[str, np.ndarray] = {}
        self.ends:   Dict[str, np.ndarray] = {}

        for lab, g in df.groupby("Label", sort=False):
            s = g["Starttime"].astype("int64").to_numpy()
            e = g["Endtime"].astype("int64").to_numpy()
            # drop empty/invalid intervals
            keep = e > s
            s, e = s[keep], e[keep]

            # sort and (optionally) merge overlaps to shrink index size
            order = np.argsort(s, kind="mergesort")
            s, e = s[order], e[order]
            # if merge_overlaps and s.size:
            #     s, e = self._merge_overlaps(s, e)

            self.starts[lab] = s
            self.ends[lab]   = e  # already sorted after merge

    def query(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        freq: str|pd.Timedelta,
        dtype: str = "int8",
        sparse: bool = False,
        labels: list = []
    ) -> pd.DataFrame:
        """
        Return a time-indexed DataFrame sampled at `freq`
        with one column per label (1 if any event of that label is active, else 0).
        """
        # left-closed, right-open grid to match interval semantics
        idx = pd.date_range(start, end, freq=freq, inclusive="left")
        t = idx.astype("int64").to_numpy()  # ns since epoch

        out = {}
        for lab in labels:
            s = self.starts.get(lab, np.empty(0, np.int64))
            e = self.ends.get(lab,   np.empty(0, np.int64))
            # count how many have started minus how many have ended → active count
            # active at t if start <= t < end  → use right for starts, left for ends
            started = np.searchsorted(s, t, side="right")
            ended   = np.searchsorted(e, t, side="left")
            active  = (started - ended) > 0
            out[lab] = active.astype(dtype)

        df = pd.DataFrame(out, index=idx)
        if sparse:
            # drastically cuts memory when events are rare
            for c in df.columns:
                df[c] = pd.arrays.SparseArray(df[c], dtype=dtype)
        return df

class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        *,
        channels: Sequence[ChannelConfig],
        patients: Sequence[str| os.PathLike],
        sample_frequency: float,
        resample_type: str = "nearest",
        total_input: str | pd.Timedelta = "30s",
        target_resolution: str | pd.Timedelta = "30s",
        event_mapping: Optional[Mapping[str, str]] = None, 
        remove_unmapped_events : bool = True, 
        get_item: Optional[Callable] = None,
        transform: Optional[list[Callable]] = None,
        num_workers:int = 4,
        online_max_tries:int = 128,
        force_one_day: bool = True
    ) -> None:
        super().__init__()
        
        # Config
        self.channels = channels
        self.event_mapping = event_mapping
        self.remove_unmapped_events = remove_unmapped_events
        self.sample_frequency = sample_frequency
        self.resample_type = resample_type
        self.total_input = pd.to_timedelta(total_input)
        self.target_resolution = pd.to_timedelta(target_resolution)
        self.get_item_callback = get_item        
        self.transform = transform
        self.all_patients = patients
        self.online_max_tries = online_max_tries
        self.initialized = False
        self.num_workers = num_workers
        self.force_one_day = force_one_day

        # Events/classes
        if event_mapping is not None:
            self.event_mapping = {k: v for k, v in event_mapping.items()}
            self.classes = sorted(list(set(self.event_mapping.values())))
        else:
            self.event_mapping = None
            self.classes = []

        if self.remove_unmapped_events and len(self.event_mapping) == 0:
            logger.warning(f"You set remove_unmapped_events to true but provided an empty mapping. If you want to not extract any labels, set event_mapping to None. If you want to extract all labels, set event_mapping to an empty dictionary and remove_unmapped_events to false.")

        # Prepared state
        self.ids = []
        self.initialize(self.all_patients, self.num_workers)

    @abstractmethod
    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        ...

    def get_extra_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        raise ValueError(f"This function should not be called")

    def has_extra_target(self) -> bool:
        return False

    def get_classes(self) -> list[str]:
        return self.classes

    def get_timeseries_len(self) -> int:
        freq = pd.to_timedelta(1.0 / self.sample_frequency, unit="s")
        return int(self.total_input.total_seconds() / freq.total_seconds())

    def get_n_patients(self) -> int:
        return len(self.edf_files)

    def __len__(self):
        return sum([f.length for f in self.edf_files])

    def prepare_patient(self, edf_path) -> Optional[EDFFile]:
        channel_names = [c.name for c in self.channels]
        normalizers = {c.name: copy.deepcopy(c.normalizer) for c in self.channels if c.normalizer is not None}

        classes = set()
        extra_classes = set()
        try:
            data_df = edf_to_df(edf_path, channel_names, start=None, end=None, frequency=self.sample_frequency, how=self.resample_type)
            if data_df is None or len(data_df) == 0: 
                raise ValueError(f"Found empty EDF file")
            
            meta = read_edf_meta(edf_path)
            start = meta["start"]
            end = meta["end"]

            start = max(data_df.index[0], start)
            end = min(data_df.index[-1], end)

            for col in normalizers.keys():
                if col in data_df.columns:
                    X = data_df[col].to_numpy(dtype=float).reshape(-1, 1)
                    normalizers[col].fit(X = X) 

            if self.event_mapping is not None:
                if self.has_extra_target():
                    df = self.get_event_df(edf_path, start) 
                    df_additional = self.get_extra_event_df(edf_path, start)
                    
                    if self.remove_unmapped_events:
                        df_additional["Label"] = df_additional["Label"].apply(lambda x: self.event_mapping[x] if x in self.event_mapping else None)
                    else:
                        df_additional["Label"] = df_additional["Label"].apply(lambda x: self.event_mapping[x] if x in self.event_mapping else x)
                    df_additional = df_additional.dropna()
                    extra_classes = set(df_additional["Label"].unique())
                    df_additional = EventIndex(df_additional)
                else:
                    df = self.get_event_df(edf_path, start) 
                    df_additional = None

                if self.remove_unmapped_events:
                    df["Label"] = df["Label"].apply(lambda x: self.event_mapping[x] if x in self.event_mapping else None)
                else:
                    df["Label"] = df["Label"].apply(lambda x: self.event_mapping[x] if x in self.event_mapping else x)
                df = df.dropna() 

                if self.force_one_day and (len(df["Starttime"].dt.date.unique()) > 2 or len(df["Endtime"].dt.date.unique()) > 2):
                    raise ValueError(f"Edf file: {edf_path} appears to be longer than one entire day. Is this a loading error? If not, set force_one_day = False")

                classes = set(df["Label"].unique())

                start = max(start, df["Starttime"].min())
                end = min(end, df["Endtime"].max())

                df = EventIndex(df)
            else:
                df = None
                df_additional = None

            n_items = int((end-self.total_input-start)/self.target_resolution)

            if n_items <= 0:
                raise ValueError(f"Edf file: {edf_path} appears to be empty between {start} - {end} with a total signal length of {end-start}s")

            return EDFFile(path=edf_path, X = None, channels=list(data_df.columns), length=n_items, labels=df, labels_extra=df_additional, start_date=start, classes=classes.union(extra_classes), normalizers=normalizers)
        except Exception as e:
            logger.warning(f"Cannot read edf file: {edf_path} due to {e}")

            return None #EDFFile(path=edf_path, classes=classes.union(extra_classes))

    def initialize(self, patients: Sequence[str | os.PathLike], num_workers: int = 4) -> None:
        """
        Initialize dataset by preparing EDF files for all (or some) patients.
        Supports parallel loading via multiprocessing.Pool with true early stop.
        """
        total_n_patients = len(patients)
        file_handles, lower_bounds, upper_bounds = [], [], []
        all_classes = set()
        lower, n_windows = 0, 0

        logger.progress_start(len(patients), desc="Preparing labels and sliding windows", leave=True)

        if num_workers > 1:
            pool = multiprocessing.Pool(num_workers)
            iter_objects = pool.imap_unordered(partial(self.prepare_patient), patients)
        else:
            iter_objects = patients

        lower = 0
        for edf in iter_objects: 
            if num_workers <= 1:
                edf = self.prepare_patient(edf)

            if edf:
                edf = cast(EDFFile, edf)
                file_handles.append(edf)
                lower_bounds.append(lower)
                upper_bounds.append(lower + edf.length)
                lower += edf.length
                n_windows += edf.length

                if edf.classes is not None:
                    all_classes |= set(edf.classes)
            
            logger.progress_advance(1)

        logger.progress_close()

        self.edf_files = file_handles
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        if not self.event_mapping:
            self.classes = sorted(list(all_classes))

        logger.info(
            f"Dataset initialized with {len(self.edf_files)}/{total_n_patients} patients. "
            f"Total windows: {len(self)}. Classes: {len(self.classes)}"
        )
        self.initialized = True

    def get_item(self, file: EDFFile, start_date: pd.Timestamp):
        # start_date:pd.Timestamp, end_date, channels, sample_frequency, resample_type)
        end_date = start_date + self.total_input
        x_df = file.get_x(start_date, end_date, self.sample_frequency, self.resample_type)
        
        # Target time at center for window/sequence modes
        t_center = x_df.index[0] + (self.total_input // 2 - self.target_resolution // 2)
        item: Dict[str, Any] = {"patient": file.path, "time": t_center}
        
        #if self.event_mapping is not None and len(self.classes) > 0:
        if file.labels:
            start_date_label = t_center
            end_date_label = start_date_label + self.target_resolution

            item["target"] = file.get_y(start_date_label, end_date_label, self.sample_frequency, self.classes) 
            if file.labels_extra:
                item["target_extra"] = file.get_y_extra(start_date_label, end_date_label, self.sample_frequency, self.classes) 

        # Make sure that x_df has exactly self.get_timeseries_len() entries. 
        # This can happen, when timestamps do not match exactly or there are inaccuracies for
        # very high sample rates.
        #   - if not enough entries: pad the last value at the end
        #   - if too many entries: take the first self.get_timeseries_len() entries
        if len(x_df) < self.get_timeseries_len():
            freq = pd.to_timedelta(1.0/self.sample_frequency, unit="s")
            freq = x_df.index.freq or pd.infer_freq(x_df.index)
            # Pad at end
            n = self.get_timeseries_len() - len(x_df)
            pad_idx = pd.date_range(start=x_df.index[-1] + freq, periods=n, freq=freq)
            pad_df = pd.DataFrame([x_df.iloc[-1].values] * n, columns=x_df.columns, index=pad_idx)
            x_df = pd.concat([x_df, pad_df])
        elif len(x_df) > self.get_timeseries_len():
            x_df = x_df.head(n = self.get_timeseries_len())

        if self.transform is not None:
            for t in self.transform:
                x_df = t(x_df)
            
        if self.get_item_callback is not None:
            item = self.get_item_callback(data=x_df, **item)
        else:
            item["data"] = torch.from_numpy(x_df.values).float()

        return item

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cnt = 0
        item = None
        last_exception = None
        while True:
            pidx = bisect.bisect_right(self.upper_bounds, idx)
            file = self.edf_files[pidx]

            new_idx = idx - self.lower_bounds[pidx] 
            cur_date = file.start_date + self.target_resolution * new_idx 
            
            try:
                item = self.get_item(file, cur_date)
            except Exception as e:
                last_exception = e
            finally:
                if cnt > self.online_max_tries or item is not None:
                    break
                
                cnt += 1
                idx = np.random.choice(range(len(self)))
        
        if cnt > self.online_max_tries or item is None:
            raise ValueError(f"Tried to get a clean item for {self.online_max_tries} tries in {self.__class__.__name__ } with no success. Last patient was {file.path}. Exception was {last_exception}")
        else:
            return item