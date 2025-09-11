from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import copy
from dataclasses import dataclass
from functools import partial
import traceback
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import tqdm

from sleepwalker.utils import logger
from sleepwalker.core.signal import Normalizer, edf_to_df, read_edf_meta
from sleepwalker.core.labels import window_labels, sequence_labels
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
    
    # Stack tensors for all keys except "timestamps"
    return {k: torch.stack(v) if k not in ignore_list else v for k, v in final_dict.items()}

import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional

class EventIndex:
    """
    Fast indicator sampling for labeled time intervals.
    Semantics: event is active at t iff start <= t < end  (left-closed, right-open).
    All timestamps should be tz-aligned (ideally UTC).
    """
    def __init__(self, events: pd.DataFrame, labels:list[str]): #merge_overlaps: bool = True
        # expected columns: starttime, endtime, label
        df = events.copy()

        self.labels: list = labels #sorted(df["Label"].unique())
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
    ) -> pd.DataFrame:
        """
        Return a time-indexed DataFrame sampled at `freq`
        with one column per label (1 if any event of that label is active, else 0).
        """
        # left-closed, right-open grid to match interval semantics
        idx = pd.date_range(start, end, freq=freq, inclusive="left")
        t = idx.astype("int64").to_numpy()  # ns since epoch

        cols = self.labels
        out = {}
        for lab in cols:
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
        sample_frequency: float,
        resample_type: str = "nearest",
        total_input: str | pd.Timedelta = "30s",
        target_resolution: str | pd.Timedelta = "30s",
        event_mapping: Optional[Mapping[str, str]] = None,
        filter_patient: Optional[Callable] = None,
        get_item: Optional[Callable] = None,
        verbose: str="TQDM",
        transform: Optional[Any] = None,
        online_filtering:bool = True
        # event_type: str = "window",
        # normalizer_fit_strategy: Optional[Mapping[str, Any]] = None,
        # num_workers: int = 0,
    ) -> None:
        super().__init__()
        
        # Config
        self.channels = channels
        self.event_mapping = event_mapping
        self.sample_frequency = sample_frequency
        self.resample_type = resample_type
        self.total_input = pd.to_timedelta(total_input)
        self.target_resolution = pd.to_timedelta(target_resolution)
        self.filter_patient = filter_patient        
        self.get_item_callback = get_item        
        self.verbose = verbose
        self.transform = transform
        self.online_filtering = online_filtering
        
        # self.num_workers = int(num_workers)
        # self.event_type = event_type
        # self.normalizer_fit_strategy = normalizer_fit_strategy or {"mode": "full"}

        # # Channels
        # self._specs = [ChannelSpec(**c) if not isinstance(c, ChannelSpec) else c for c in channels]

        # # Events/classes
        if event_mapping is not None:
            self.event_mapping = {k.lower(): v.lower() for k, v in event_mapping.items()}
            self.classes = sorted(list(set(self.event_mapping.values())))
        else:
            self.event_mapping = None
            self.classes = []

        # Prepared state
        self.ids = []
        # self._patients: List[str] = []
        # self._extractors: Dict[str, SignalExtractor] = {}
        # self._events: Dict[str, pd.DataFrame] = {}
        # self._ids: List[Tuple[str, pd.Timestamp]] = []
        # self._class_index: Optional[np.ndarray] = None
        # Always call prepare(patients) explicitly after constructing

    @abstractmethod
    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_extra_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        ...

    @abstractmethod
    def has_extra_target(self) -> bool:
        ...

    def get_classes(self) -> list[str]:
        return self.classes

    def get_timeseries_len(self) -> int:
        freq = pd.to_timedelta(1.0 / self.sample_frequency, unit="s")
        return self.total_input * freq # TODO FROM HERE

    def __len__(self):
        return len(self.ids)

    def get_events(self, df, start_date):
        start_date += self.total_input // 2 - self.target_resolution // 2
        end_date = start_date + self.target_resolution

        freq = pd.to_timedelta(1.0 / self.sample_frequency, unit="s")
        return df.query(start_date, end_date, freq=freq, sparse=False)

        # time_index = pd.date_range(start=start_date, end=end_date, freq=freq)

        # onehot_df = pd.DataFrame(0, index=time_index, columns=self.classes, dtype=int)

        # dff = df[(df["Starttime"] >= start_date) & (df["Endtime"] <= end_date)]
        # for _, row in dff.iterrows():
        #     label = row["Label"]
        #     if label not in self.classes:
        #         continue
        #     event_start = row["Starttime"]
        #     event_end = row["Endtime"]
        #     # Set 1 for all time points within the event interval
        #     mask = (onehot_df.index >= event_start) & (onehot_df.index < event_end)
        #     onehot_df.loc[mask, label] = 1

        # return onehot_df

        # target_window_is_middle
        #duration_by_label = dff.groupby("Label")["Duration"].sum().to_dict()
        #return torch.tensor([duration_by_label.get(c, 0.0) for c in self.classes])
        # dff_sum = dff.sum(axis=0) * self.target_resolution.total_seconds()
        # return torch.tensor(dff_sum[self.classes].values)

    def prepare_patient(self, edf_path) -> Tuple[str, list[int], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[dict], int]:
        ids = []
        n_skipped = 0
        channel_names = [c.name for c in self.channels]
        normalizers = {c.name: copy.deepcopy(c.normalizer) for c in self.channels if c.normalizer is not None}

        try:
            if self.filter_patient is not None and not self.filter_patient(edf_path):
                return edf_path, ids, None, None, None, n_skipped
            
            data_df = edf_to_df(edf_path, channel_names, start=None, end=None, frequency=self.sample_frequency, how=self.resample_type)
            if data_df is None:
                return edf_path, ids, None, None, None, n_skipped
            
            meta = read_edf_meta(edf_path)
            start = meta["start"]
            end = meta["end"]

            for col in normalizers.keys():
                if col in data_df.columns:
                    X = data_df[col].to_numpy(dtype=float).reshape(-1, 1)
                    normalizers[col].fit(X, self.sample_frequency) 

            if self.event_mapping is not None:
                if self.has_extra_target():
                    df = self.get_event_df(edf_path, start) 
                    df_additional = self.get_extra_event_df(edf_path, start)
                    
                    df_additional["Label"] = df_additional["Label"].apply(lambda x: self.event_mapping[x] if x in self.event_mapping else None)
                    df_additional = df_additional.dropna()
                    df_additional = EventIndex(df_additional, self.classes)
                else:
                    df = self.get_event_df(edf_path, start) 
                    df_additional = None

                df["Label"] = df["Label"].apply(lambda x: self.event_mapping[x] if x in self.event_mapping else None)
                df = df.dropna() 
                start = max(start, df["Starttime"].min())
                end = min(end, df["Endtime"].max())

                df = EventIndex(df, self.classes)
            else:
                df = None
                df_additional = None
            
            if self.get_item_callback is not None and not self.online_filtering:
                while start < end-self.total_input:
                    if self.get_item_callback is not None:
                        item = self.get_item(edf_path, start, normalizers)
                        if item is not None:
                            ids.append((edf_path, start))
                        else:
                            n_skipped += 1
                    else:
                        ids.append((edf_path, start))
                    start += self.total_input
            else:
                ids = [(edf_path, t) for t in pd.date_range(start, end - self.total_input, freq=self.total_input)]

            if len(ids) == 0 and self.verbose in ["TQDM", "tqdm", "console"]:
                logger.warning(f"Edf file: {edf_path} appears to be empty between {start} - {end} with a total signal length of {end-start}s")

            return edf_path, ids, df, df_additional, normalizers, n_skipped # type: ignore
        #except (OSError, ValueError, KeyError) as e:
        except Exception as e:
            if self.verbose in ["TQDM", "tqdm", "console"]:
                logger.warning(f"Cannot read edf file: {edf_path} due to {e}")
                logger.warning(traceback.format_exc())
                # logger.warning(f"Cannot read edf file: {fpath} due to {e}")

            return edf_path, ids, None, None, None, n_skipped # type: ignore

    def initialize(self, patients: Sequence[str], num_workers: int=4) -> None:
        events = {}
        events_additional = {}
        
        all_ids = []
        all_normalizers = {}
        n_patients = 0
        n_skipped_total = 0
        total_n_patients = len(patients)
        final_patients = [] 

        if num_workers > 1:
            pool = multiprocessing.Pool(num_workers)
            iter_objects = pool.imap_unordered(partial(self.prepare_patient), patients)
        else:
            iter_objects = patients

        for ret_value in tqdm.tqdm(iter_objects, total=len(patients), desc=logger.log_prefix() + f"Preparing labels and sliding windows", disable=self.verbose not in ["TQDM", "tqdm"]): # type: ignore
            if num_workers > 1:
                fpath, ids, df, df_additional, normalizers, n_skipped = ret_value
            else:
                fpath, ids, df, df_additional, normalizers, n_skipped = self.prepare_patient(ret_value)
            
            if len(ids) > 0:
                all_ids.extend(ids)
                events[fpath] = df
                all_normalizers[fpath] = normalizers
                events_additional[fpath] = df_additional
                n_patients += 1
                n_skipped_total += n_skipped # type: ignore
                final_patients.append(fpath)

        if num_workers > 1:
            pool.close() # type: ignore
            pool.join() # type: ignore
        
        self.patients = final_patients        
        self.ids = all_ids
        self.events = events
        self.events_additional = events_additional
        self.n_patients = n_patients
        self.all_normalizers = all_normalizers
        if self.verbose in ["TQDM", "tqdm", "console"]:
            logger.info(f"Dataset initialized with {self.n_patients}/{total_n_patients} patients. Skipped {n_skipped_total} windows due to insufficient labels. There are {len(self.ids)} windows remaining. ")

    def get_item(self, edf_path:str, start_date:pd.Timestamp, normalizers:Optional[dict[str,Normalizer]]):
        end_date = start_date + self.total_input
        channel_names = [c.name for c in self.channels]
        x_df = edf_to_df(edf_path, channel_names, start_date, end_date, self.sample_frequency, self.resample_type)
        
        # and normalizers.get(col) is not None:
        if normalizers:
            for col, norm in normalizers.items():
                if col in x_df.columns:
                    vals = x_df[col].to_numpy(dtype=float).reshape(-1, 1)
                    x_df[col] = norm.transform(vals, self.sample_frequency).ravel()

        # Pad to exact desired length if needed
        # freq = pd.to_timedelta(1.0/self.sample_frequency, unit="s")
        # desired_len = int(self.total_input / freq)
        # if len(x_df) < desired_len:
        #     freq = x_df.index.freq or pd.infer_freq(x_df.index)
        #     # Pad at end
        #     n = desired_len - len(x_df)
        #     pad_idx = pd.date_range(start=x_df.index[-1] + freq, periods=n, freq=freq)
        #     pad_df = pd.DataFrame([x_df.iloc[-1].values] * n, columns=x_df.columns, index=pad_idx)
        #     x_df = pd.concat([x_df, pad_df])

        X = torch.from_numpy(x_df.values).float()
        if self.transform is not None:
            X = self.transform(X)

        # Target time at center for window/sequence modes
        t_center = x_df.index[0] + (self.total_input // 2 - self.target_resolution // 2)
        item: Dict[str, Any] = {"patient": edf_path, "time": t_center, "data": X}
        
        if self.event_mapping is not None and len(self.classes) > 0:
            y_df = self.events[edf_path]
            item["target"] = self.get_events(y_df, start_date)
        
            if self.has_extra_target():
                additional_df = self.events_additional[edf_path]
                item["target_extra"] = self.get_events(additional_df, start_date)

        if self.get_item_callback is not None:
            return self.get_item_callback(item)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        edf_path, start_date = self.ids[idx]
        normalizer = self.all_normalizers[edf_path]
        item = self.get_item(edf_path, start_date, normalizer)

        cnt = 0
        while item is None:
            idx = np.random.choice(range(len(self.ids)))
            edf_path, start_date = self.ids[idx]
            normalizer = self.all_normalizers[edf_path]
            
            item = self.get_item(edf_path, start_date, normalizer)
            cnt += 1
            if cnt > 50:
                raise ValueError("Tried to get a clean item for 50 tries, no success.")
        return item