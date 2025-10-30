from __future__ import annotations

from abc import ABC, abstractmethod
import bisect
from collections import Counter, defaultdict
import copy
from dataclasses import dataclass
from functools import partial
import os
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sleepwalker.utils import logger
from sleepwalker.core.signal import edf_to_df, read_edf_meta
from sleepwalker.datasets.normalizer import Normalizer
import multiprocessing

@dataclass
class ChannelConfig:
    name: str
    normalizer: Optional[Normalizer] = None  # externally created (may be None)

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

    # @staticmethod
    # def _merge_overlaps(s: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    #     ms, me = [s[0]], [e[0]]
    #     for i in range(1, s.size):
    #         if s[i] <= me[-1]:           # overlaps or touches (since end is exclusive this is fine)
    #             if e[i] > me[-1]:
    #                 me[-1] = e[i]
    #         else:
    #             ms.append(s[i]); me.append(e[i])
    #     return np.asarray(ms, dtype=np.int64), np.asarray(me, dtype=np.int64)

    def query(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        freq: str,
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
        transform: Optional[Any] = None,
        # online_filtering:bool = True,
        cache_patients:int = 0,
        num_workers:int = 4,
        online_max_tries:int = 128
        # event_type: str = "window",
        # normalizer_fit_strategy: Optional[Mapping[str, Any]] = None,
        # num_workers: int = 0,
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
        self.cache_patients = cache_patients
        self.cache = {}

        self.online_max_tries = online_max_tries
        self.initialized = False

        # # Events/classes
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
        self.initialize(patients, num_workers)
        # self._patients: List[str] = []
        # self._extractors: Dict[str, SignalExtractor] = {}
        # self._events: Dict[str, pd.DataFrame] = {}
        # self._ids: List[Tuple[str, pd.Timestamp]] = []
        # self._class_index: Optional[np.ndarray] = None
        # Always call prepare(patients) explicitly after constructing

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
        return len(self.patients)

    def __len__(self):
        return sum([p[3] for p in self.patients])

    def get_events(self, df, start_date):
        start_date += self.total_input // 2 - self.target_resolution // 2
        end_date = start_date + self.target_resolution

        freq = pd.to_timedelta(1.0 / self.sample_frequency, unit="s")
        return df.query(start_date, end_date, freq=freq, sparse=False, labels=self.classes)

    def prepare_patient(self, edf_path) -> Tuple[str, int, Optional[EventIndex], Optional[EventIndex], Optional[dict], Optional[pd.Timestamp], Set[str]]:
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

            # TODO If caching, then transform data inplace and return -> 

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
                classes = set(df["Label"].unique())

                start = max(start, df["Starttime"].min())
                end = min(end, df["Endtime"].max())

                df = EventIndex(df)
            else:
                df = None
                df_additional = None

            n_items = int((end-self.total_input-start)/self.target_resolution)

            if n_items == 0:
                raise ValueError(f"Edf file: {edf_path} appears to be empty between {start} - {end} with a total signal length of {end-start}s")
                
            return edf_path, n_items, df, df_additional, normalizers, start, classes.union(extra_classes) 

            # if self.get_item_callback is not None and not self.online_filtering:
            #     while start < end-self.total_input:
            #         if self.get_item_callback is not None:
            #             item = self.get_item(edf_path, start, normalizers)
            #             if item is not None:
            #                 ids.append((edf_path, start))
            #             else:
            #                 n_skipped += 1
            #         else:
            #             ids.append((edf_path, start))
            #         start += self.target_resolution

            #     if len(ids) == 0:
            #         raise ValueError(f"Edf file: {edf_path} appears to be empty between {start} - {end} with a total signal length of {end-start}s")    
            #     return edf_path, ids, df, df_additional, normalizers, n_skipped, classes.union(extra_classes) # type: ignore
            # else:
            #     n_items = int((end-self.total_input).total_seconds()/self.target_resolution.total_seconds())
            #     ids = [(edf_path, t) for t in pd.date_range(start, end - self.total_input, freq=self.target_resolution)]

            #     if n_items == 0:
            #         raise ValueError(f"Edf file: {edf_path} appears to be empty between {start} - {end} with a total signal length of {end-start}s")
                 
            #     return edf_path, n_items, df, df_additional, normalizers, n_skipped, classes.union(extra_classes) # type: ignore
        #except (OSError, ValueError, KeyError) as e:
        except Exception as e:
            logger.warning(f"Cannot read edf file: {edf_path} due to {e}")
            # logger.warning(traceback.format_exc())
            # logger.warning(f"Cannot read edf file: {fpath} due to {e}")

            return edf_path, 0, None, None, None, None, classes.union(extra_classes) # type: ignore

    def initialize(self, patients: Sequence[str|os.PathLike], num_workers: int=4) -> None:
        if self.cache_patients > 0:
            rng = np.random.default_rng()
            patients = rng.choice(patients, size=self.cache_patients, replace=False)

        events = {}
        events_additional = {}
        
        upper_bounds = []
        final_patients = []

        all_normalizers = {}
        all_classes = set()
        total_n_patients = len(patients)

        if num_workers > 1:
            pool = multiprocessing.Pool(num_workers)
            iter_objects = pool.imap_unordered(partial(self.prepare_patient), patients)
        else:
            iter_objects = patients

        logger.progress_start(len(patients), desc="Preparing labels and sliding windows", leave=True)

        lower = 0
        for ret_value in iter_objects: 
            if num_workers > 1:
                fpath, n_items, df, df_additional, normalizers, start_date, classes = ret_value
            else:
                fpath, n_items, df, df_additional, normalizers, start_date, classes = self.prepare_patient(ret_value)
            
            # if len(ids) > 0:
            if n_items > 0:
                final_patients.append( (fpath, lower, start_date, n_items) )
                upper_bounds.append(lower + n_items)
                lower += n_items

                events[fpath] = df
                all_normalizers[fpath] = normalizers
                events_additional[fpath] = df_additional
                # n_patients += 1
                # n_skipped_total += n_skipped # type: ignore
                all_classes = all_classes.union(classes)
            logger.progress_advance(1)
        
        logger.progress_close()

        if num_workers > 1:
            pool.close() # type: ignore
            pool.join() # type: ignore
        
        self.patients = final_patients        
        self.upper_bounds = upper_bounds
        self.events = events
        self.events_additional = events_additional
        # self.n_patients = n_patients
        self.all_normalizers = all_normalizers
        if len(self.event_mapping) == 0:
            self.classes = sorted(list(all_classes))

        #  Skipped {n_skipped_total} windows due to insufficient labels. There are {len(self.ids)} windows remaining. 
        logger.info(f"Dataset initialized with {len(self.patients)}/{total_n_patients} patients.There are {len(self)} windows available.")
        self.initialized = True

    def get_item(self, edf_path:str, start_date:pd.Timestamp, normalizers:Optional[dict[str,Normalizer]]):
        end_date = start_date + self.total_input
        channel_names = [c.name for c in self.channels]
        x_df = edf_to_df(edf_path, channel_names, start_date, end_date, self.sample_frequency, self.resample_type)
        
        # and normalizers.get(col) is not None:
        if normalizers:
            for col, norm in normalizers.items():
                if col in x_df.columns:
                    vals = x_df[col].to_numpy(dtype=float).reshape(-1, 1)
                    x_df[col] = norm.transform(vals).ravel()

        # Target time at center for window/sequence modes
        t_center = x_df.index[0] + (self.total_input // 2 - self.target_resolution // 2)
        item: Dict[str, Any] = {"patient": edf_path, "time": t_center}
        
        if self.event_mapping is not None and len(self.classes) > 0:
            y_df = self.events[edf_path]
            item["target"] = self.get_events(y_df, start_date)
        
            if self.has_extra_target():
                additional_df = self.events_additional[edf_path]
                item["target_extra"] = self.get_events(additional_df, start_date)

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

        if self.get_item_callback is not None:
            item = self.get_item_callback(data=x_df, **item)
        else:
            item["data"] = torch.from_numpy(x_df.values).float()

        if self.transform is not None:
            item["data"] = self.transform(item["data"])
        return item

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cnt = 0
        item = None

        while True:
            pidx = bisect.bisect_right(self.upper_bounds, idx)
            edf_path, lower, start_date, n_items = self.patients[pidx]
            new_idx = idx - lower 
            cur_date = start_date + self.target_resolution * new_idx 
            normalizer = self.all_normalizers[edf_path]
            
            try:
                item = self.get_item(edf_path, cur_date, normalizer)
            except Exception as e:
                pass
            finally:
                if cnt > self.online_max_tries or item is not None:
                    break

                cnt += 1
                idx = np.random.choice(range(len(self)))
        
        if cnt > self.online_max_tries or item is None:
            raise ValueError(f"Tried to get a clean item for {self.online_max_tries} tries in {self.__class__.__name__ } with no success. Last patient was {edf_path}")
        else:
            return item