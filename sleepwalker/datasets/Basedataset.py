"""Core dataset abstractions for EDF-backed Sleepwalker workflows.

This module defines the base dataset lifecycle used throughout the repository:
configure channels and label mappings, prepare patient-level metadata, build a
sliding-window index, and lazily materialize model-ready samples on demand.

The code is used by dataset adapters under :mod:`sleepwalker.datasets`, by
training scripts, and by deployment code that exports unlabelled dataset
templates for later inference. Tests in ``tests/test_datasets.py`` and
``tests/test_deployment.py`` exercise the window-building, lazy loading, and
export-related behavior documented here.
"""

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
    """Describe one requested signal channel for dataset loading.

    Args:
        name: Channel name expected in the EDF file.
        normalizer: Optional normalizer fitted per patient and applied when the
            corresponding signal is loaded.
        group: Optional conceptual group name. When multiple configured
            channels share a group, one available channel is sampled per item
            and renamed to the group name.
        quality_name: Optional companion channel used as per-window quality
            metadata. When present and selected, it is passed to
            ``prepare_sample`` as ``quality_data``.
    """

    name: str
    normalizer: Optional[Normalizer] = None  
    group: Optional[str] = None
    quality_name: Optional[str] = None

@dataclass
class EDFFile: 
    """Prepared patient descriptor used after dataset initialization.

    The object stores lightweight metadata plus lazily queried label indices.
    Signal values are normally re-read from disk in :meth:`get_x` rather than
    kept resident after initialization.
    """

    channels: List[str]
    path: str
    start_date: pd.Timestamp
    start_offsets: Optional[np.ndarray] = None
    handle: Optional[pyedflib.EdfReader] = None
    length: int = 0
    X: Optional[pd.DataFrame] = None
    classes: Optional[Set[str]] = None
    labels: Optional[EventIndex] = None
    labels_extra: Optional[EventIndex] = None
    normalizers: Optional[dict[str, Normalizer]] = None

    def get_x(self, start_date:pd.Timestamp, end_date:pd.Timestamp, sample_frequency, resample_type):
        """Load one signal window for the prepared patient.

        Args:
            start_date: Inclusive window start.
            end_date: Inclusive or near-inclusive window end as passed to
                ``edf_to_df``.
            sample_frequency: Requested resampling frequency in Hz.
            resample_type: Resampling mode forwarded to ``edf_to_df``.

        Returns:
            A DataFrame indexed by timestamps and containing the configured
            signal columns. If patient-level normalizers were fitted during
            preparation, they are applied column-wise before returning.
        """
        if self.X is None:
            x_df = edf_to_df(self.path, self.channels, start_date, end_date, sample_frequency, resample_type, True)
            # TODO allow normalization after augmentation?  
            if self.normalizers:
                for col, norm in self.normalizers.items():
                    if col in x_df.columns:
                        vals = x_df[col].to_numpy(dtype=float).reshape(-1, 1)
                        x_df[col] = norm.transform(vals).ravel()

            return x_df
        else:
            return self.X.loc[start_date:end_date]
    
    def get_y_extra(self, start_date: pd.Timestamp, end_date: pd.Timestamp, sample_frequency, classes):
        """Sample the optional secondary target timeline for one window."""
        if self.labels_extra:
            freq = pd.to_timedelta(1.0 / sample_frequency, unit="s")
            return self.labels_extra.query(start_date, end_date, freq=freq, sparse=False, labels=classes)
        else:
            return None

    def get_y(self, start_date: pd.Timestamp, end_date: pd.Timestamp, sample_frequency, classes):
        """Sample the primary target timeline for one window."""
        if self.labels:
            freq = pd.to_timedelta(1.0 / sample_frequency, unit="s")
            return self.labels.query(start_date, end_date, freq=freq, sparse=False, labels=classes)
        else:
            return None

def batch_collate(batch, ignore_list = ["time", "patient"]):
    """Stack tensor-like batch fields while preserving metadata lists.

    Args:
        batch: Sequence of per-item dictionaries returned by the dataset.
        ignore_list: Keys that should stay as Python lists instead of being
            stacked with ``torch.stack``.

    Returns:
        A dictionary with one entry per observed key. Tensor-valued fields are
        stacked, while metadata such as ``time`` and ``patient`` remains a
        list.
    """
    final_dict = defaultdict(list)

    for b in batch:
        if b:
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
        """Sample active labels on a regular time grid.

        Args:
            start: Window start timestamp.
            end: Window end timestamp.
            freq: Sampling frequency expressed as a pandas offset or timedelta.
            dtype: Output dtype for dense arrays.
            sparse: Whether to use pandas sparse arrays for the result.
            labels: Labels to materialize as output columns.

        Returns:
            A time-indexed DataFrame with one column per requested label. Each
            value is ``1`` when any interval with that label is active at the
            sample time and ``0`` otherwise.

        Notes:
            Interval semantics are left-closed and right-open:
            ``start <= t < end``.
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
    """
    Base class for EDF-backed datasets that turn raw patient recordings into
    model-ready training items.

    In practice, a `BaseDataset` does three jobs for you:

    1. load and resample signals from EDF files
    2. load and map raw event labels to task labels
    3. expose sliding-window items that contain at least `data`, `target`,
       `patient`, and `time`

    The class is designed for research workflows where different experiments
    need slightly different filtering or target-building logic without having to
    reimplement EDF loading every time.

    Typical usage
    -------------
    1. Configure a dataset object with channels, label mapping, and optional
       preparation callbacks.
    2. Optionally inspect patients with `get_patient_stats(...)` if you want to
       filter patients before training.
    3. Call `initialize(patients, num_workers)` once you know which patients to
       keep.
    4. Use the dataset with a `DataLoader`.

    The dataset is intentionally **not initialized automatically**. This allows
    you to:
    - inspect/filter patients before the expensive sliding-window index is built
    - reuse one configured dataset object for both patient statistics and final training

    Loading model and performance
    -----------------------------
    `BaseDataset` is designed for large EDF collections. It does **not** keep
    full patient signals in memory after initialization.

    Instead, the workflow is:
    - `initialize(...)` scans the selected patients, prepares labels, and builds
      an index of valid sliding-window positions
    - `__getitem__` uses that index to choose a candidate item
    - the corresponding EDF signal window is loaded lazily only when needed

    This means you can work with very large sets of EDF files without large
    memory overhead. The main bottleneck is usually disk / network I/O and EDF
    parsing, not RAM usage.

    Item retrieval fundamentally uses rejection sampling:
    - a candidate window is selected from the precomputed index
    - `prepare_target` may reject it before signal loading
    - `prepare_sample` may reject it after signal loading
    - if rejected, another candidate is sampled until a valid item is found or
      `online_max_tries` is exceeded

    In practice, performance depends strongly on where you place filtering:
    - filtering in `prepare_target` is cheap and usually preferable
    - filtering in `prepare_sample` is more expensive because signal I/O has
      already happened
    - expensive patient-wide checks in `prepare_patient` run only once per
      patient, but they still need the full patient signal to be loaded during
      preparation

    Sparse window indexing
    ----------------------
    After `prepare_patient` returns its possibly filtered `label_df`,
    `BaseDataset` no longer assumes that valid windows form one continuous
    `[start, end]` region. Instead, it scans the remaining label intervals for
    timestamp discontinuities and builds a sparse per-patient index of valid
    stride offsets.

    This means user code can trim labels down to several disjoint retained
    regions and the dataset will sample only windows fully contained in those
    connected components, instead of densely sampling the gaps in between.

    Preparation hooks
    -----------------
    There are three optional hooks. They are ordered from expensive to cheap to
    help you place logic in the right stage.

    `prepare_patient`
        Runs once per patient after the full patient signal and mapped labels
        have been loaded.

        Use this for whole-patient logic such as:
        - trimming leading/trailing wake
        - rejecting patients with too few valid labels
        - rejecting patients with obviously broken signals

        This is the most expensive hook because it sees the full loaded signal.
        Only put logic here that really needs patient-wide context.

    `prepare_target`
        Runs once per candidate item before the signal window is loaded. It
        receives the target interval labels and should either:
        - return `None` to reject the item cheaply, or
        - return a dictionary with the prepared target payload

        Use this for:
        - cheap label-based filtering
        - multiclass / multilabel target construction
        - class-balancing metadata

        If your logic can be expressed here, this is usually the best place for
        it because no signal window has to be loaded yet.

    `prepare_sample`
        Runs after the signal window has been loaded, re-referenced and resized.
        It should return the final sample dictionary or `None` to reject the
        sample.

        Use this for:
        - converting the signal DataFrame to tensors
        - signal-dependent rejection, e.g. flatline or NaN-heavy windows
        - attaching final `data` / `target` outputs

        This is more expensive than `prepare_target` because the signal window
        has already been loaded.

    Which hook should I use?
    ------------------------
    Use `prepare_patient` when the decision depends on the full patient.

    Example:
    - trim wake at the start/end of the night
    - reject patients with almost no sleep left after trimming

    Use `prepare_target` when the decision depends only on labels or timestamps
    for one candidate item.

    Example:
    - reject windows outside sleep
    - build one-hot multiclass targets

    Use `prepare_sample` when the decision depends on the loaded signal window.

    Example:
    - reject windows with too many NaNs
    - convert signal frames to tensors

    If you want to compare patients to each other, for example “drop the
    shortest sleepers” or “keep only the top-K by REM time”, use
    `get_patient_stats(...)` before calling `initialize(...)`.

    Parameters
    ----------
    channels
        Sequence of `ChannelConfig` objects describing which EDF channels to
        load. If multiple channels share the same `group`, one representative is
        sampled per group in grouped-channel settings.

    sample_frequency
        Target sampling frequency in Hz used when loading signal windows and
        querying label timelines.

    resample_type
        Resampling mode passed to `edf_to_df(...)`. This controls how signals
        are aligned to `sample_frequency`. Common choices depend on the signal
        loader implementation; `"nearest"` is the default and safe for most
        annotation-aligned use cases.

    total_input
        Length of the signal window returned for each item. Accepts values such
        as `"30s"`, `"5min"`, or a `pd.Timedelta`.

    target_resolution
        Length of the target interval associated with each item. In sleep
        staging this is often `"30s"`.

    stride
        Time between consecutive sampled windows. Defaults to
        `target_resolution`, which preserves the previous behavior. Set
        `stride < total_input` to sample overlapping windows while keeping the
        target interval length unchanged.

    event_mapping
        Mapping from raw dataset-specific event labels to the labels used by
        your task, e.g. `{"Sleep stage W": "wake"}`.

    remove_unmapped_events
        If `True`, labels that are not found in `event_mapping` are dropped. If
        `False`, unmapped labels are kept as-is.

    prepare_patient
        Optional callback for whole-patient preparation. It receives
        `data_df`, `label_df`, `label_extra_df`, and `patient`.
        It should return `(data_df, label_df, label_extra_df)` or `None`.

    prepare_target
        Optional callback for per-item target preparation. It receives
        `target`, `target_extra`, `patient`, and `time`. It should return a
        dictionary or `None`.

    prepare_sample
        Optional callback for final sample preparation. It receives the loaded
        signal window as `data` plus the current item fields. It should return
        the final item dictionary or `None`.

    online_max_tries
        Maximum number of times `__getitem__` retries random alternative windows
        when a sampled item gets rejected by `prepare_target` or
        `prepare_sample`.

    force_one_day
        If `True`, reject patients whose mapped labels appear to span more than
        one day. This catches some common loading/pathology issues.

    rereference
        Optional list of channel groups that should be average-referenced after
        the signal window is loaded.

        Example:
        `rereference=[["C3", "C4"]]`
        means both channels are replaced by their values minus the mean of
        `C3`/`C4` at each time step.

    Examples
    --------
    Trim wake once per patient:

    ```python
    def prepare_patient(data_df, label_df, label_extra_df, patient=None):
        trimmed = trim_event(data_df, label_df, label_extra_df)
        if trimmed is None:
            return None
        label_df, label_extra_df = trimmed
        return data_df, label_df, label_extra_df
    ```

    Build a multiclass target and reject low-sleep windows cheaply:

    ```python
    from functools import partial
    from sleepwalker.trainer.utils.targets import prepare_multiclass_target

    prepare_target = partial(
        prepare_multiclass_target,
        target_classes=["wake", "n1", "n2", "n3", "rem"],
        filters=[{"columns": ["n1", "n2", "n3", "rem"], "percentage": 0.5, "mode": "min"}],
    )
    ```

    Convert signals to tensors and reject bad windows:

    ```python
    def prepare_sample(data, target, target_extra=None, patient=None, time=None):
        if data.isna().mean().mean() > 0.1:
            return None
        item = {
            "data": torch.from_numpy(data.values).float(),
            "target": target,
            "patient": patient,
            "time": time,
        }
        if target_extra is not None:
            item["target_extra"] = target_extra
        return item
    ```
    """
    def __init__(
        self,
        *,
        channels: Sequence[ChannelConfig],
        sample_frequency: float,
        resample_type: str = "nearest",
        total_input: str | pd.Timedelta = "30s",
        target_resolution: str | pd.Timedelta = "30s",
        stride: Optional[str | pd.Timedelta] = None,
        event_mapping: Optional[Mapping[str, str]] = None, 
        remove_unmapped_events : bool = True, 
        prepare_patient: Optional[Callable[[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]], Optional[tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]]]] = None,
        prepare_target: Optional[Callable] = None,
        prepare_sample: Optional[Callable] = None,
        online_max_tries:int = 128,
        force_one_day: bool = True,
        rereference: Optional[List[List[str]]] = None, # [ ["C3-A1", "C4-A2"] ]
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
        self.stride = pd.to_timedelta(stride) if stride is not None else self.target_resolution
        if self.stride <= pd.Timedelta(0):
            raise ValueError("stride must be positive.")
        if self.stride > self.total_input:
            raise ValueError("stride must be smaller than or equal to total_input.")
        self.prepare_target_callback = prepare_target
        self.prepare_sample_callback = prepare_sample
        self.prepare_patient_callback = prepare_patient
        self.all_patients: list[str | os.PathLike] = []
        self.online_max_tries = online_max_tries
        # self.online_retry_scope = "global"
        self.initialized = False
        self.force_one_day = force_one_day
        self.rereference = rereference
        self.edf_files: list[EDFFile] = []
        self.lower_bounds: list[int] = []
        self.upper_bounds: list[int] = []
        self.channel_groups: Dict[str, list[str]] = defaultdict(list)
        self.channel_configs_by_name: Dict[str, ChannelConfig] = {}
        self.channel_configs_by_group: Dict[str, list[ChannelConfig]] = defaultdict(list)
        for cfg in self.channels:
            self.channel_configs_by_name[cfg.name] = cfg
            if cfg.group is not None:
                self.channel_groups[cfg.group].append(cfg.name)
                self.channel_configs_by_group[cfg.group].append(cfg)
            else:
                self.channel_groups[cfg.name].append(cfg.name)
                self.channel_configs_by_group[cfg.name].append(cfg)

        # Events/classes
        if event_mapping is not None:
            self.event_mapping = {k: v for k, v in event_mapping.items()}
            self.classes = sorted(list(set(self.event_mapping.values())))
            self.label_classes = list(self.classes)
        else:
            self.event_mapping = None
            self.classes = []
            self.label_classes = []

        if self.event_mapping is not None and self.remove_unmapped_events and len(self.event_mapping) == 0:
            logger.warning(f"You set remove_unmapped_events to true but provided an empty mapping. If you want to not extract any labels, set event_mapping to None. If you want to extract all labels, set event_mapping to an empty dictionary and remove_unmapped_events to false.")

    @abstractmethod
    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        """Return raw event intervals for one patient recording.

        Args:
            edf_path: Path to the patient EDF file.
            start_datetime: Recording start time inferred from EDF metadata.

        Returns:
            A DataFrame containing at least ``Starttime``, ``Endtime``, and
            ``Label`` columns.

        Notes:
            Concrete dataset adapters must implement this method.
        """
        ...

    def get_extra_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        """Return optional secondary event intervals.

        Notes:
            The default implementation signals that extra targets are not
            supported for this dataset.
        """
        raise ValueError(f"This function should not be called")

    def has_extra_target(self) -> bool:
        """Report whether this dataset exposes a secondary target timeline."""
        return False

    def get_classes(self) -> list[str]:
        """Return the canonical class labels currently known to the dataset."""
        return self.classes

    def get_timeseries_len(self) -> int:
        """Return the expected number of signal samples per item."""
        freq = pd.to_timedelta(1.0 / self.sample_frequency, unit="s")
        return int(self.total_input.total_seconds() / freq.total_seconds())

    def get_n_patients(self) -> int:
        """Return the number of prepared patient recordings."""
        return len(self.edf_files)

    def get_input_channels(self) -> list[str]:
        """Return the effective model input channel names after grouping."""
        return list(self.channel_groups.keys())

    def to_unlabelled(self):
        """Build an inference-time dataset template without labels.

        Returns:
            An ``UnlabelledDataset`` configured with the same signal-loading and
            sample-building settings as this dataset.
        """
        from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset

        return UnlabelledDataset(
            channels=self.channels,
            sample_frequency=self.sample_frequency,
            resample_type=self.resample_type,
            total_input=self.total_input,
            stride=self.stride,
            prepare_patient=self.prepare_patient_callback,
            prepare_sample=self.prepare_sample_callback,
            online_max_tries=self.online_max_tries,
            force_one_day=self.force_one_day,
            rereference=self.rereference,
        )

    def __len__(self):
        return sum([f.length for f in self.edf_files])

    def _build_connected_segments(self, label_df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Collapse filtered labels into connected timestamp segments.

        The function treats intervals as connected when they overlap in time.
        It intentionally does not merge merely nearby intervals with a gap.
        """
        if len(label_df) == 0:
            return []

        df = label_df.sort_values(["Starttime", "Endtime"]).reset_index(drop=True)

        segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        seg_start = cast(pd.Timestamp, df.iloc[0]["Starttime"])
        seg_end = cast(pd.Timestamp, df.iloc[0]["Endtime"])

        for i in range(1, len(df)):
            row_start = cast(pd.Timestamp, df.iloc[i]["Starttime"])
            row_end = cast(pd.Timestamp, df.iloc[i]["Endtime"])
            if row_start <= seg_end:
                seg_end = max(seg_end, row_end)
            else:
                segments.append((seg_start, seg_end))
                seg_start = row_start
                seg_end = row_end

        segments.append((seg_start, seg_end))
        return segments

    def _segment_to_offsets(
        self,
        base_start: pd.Timestamp,
        seg_start: pd.Timestamp,
        seg_end: pd.Timestamp,
    ) -> np.ndarray:
        """Convert one connected time segment into candidate stride offsets.

        The filtered ``label_df`` defines where targets may be valid. We
        therefore keep windows whose target interval can overlap the retained
        segment, rather than requiring the full input context to fit inside the
        segment as well.
        """
        target_offset = self.total_input // 2 - self.target_resolution // 2
        earliest_valid_start = seg_start - target_offset - self.target_resolution
        latest_valid_start = seg_end - target_offset
        if latest_valid_start < earliest_valid_start:
            return np.empty(0, dtype=np.int64)

        first_offset = max(0, int(np.ceil((earliest_valid_start - base_start) / self.stride)))
        last_offset = int(np.floor((latest_valid_start - base_start) / self.stride))
        if last_offset < first_offset:
            return np.empty(0, dtype=np.int64)

        return np.arange(first_offset, last_offset + 1, dtype=np.int64)

    def _build_start_offsets(
        self,
        base_start: pd.Timestamp,
        label_df: pd.DataFrame,
    ) -> np.ndarray:
        """Build sparse valid window starts from the filtered patient labels."""
        segments = self._build_connected_segments(label_df)
        offsets = [
            self._segment_to_offsets(base_start=base_start, seg_start=seg_start, seg_end=seg_end)
            for seg_start, seg_end in segments
        ]
        offsets = [arr for arr in offsets if len(arr) > 0]
        if len(offsets) == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(offsets)

    def _prepare_patient_artifacts(self, edf_path):
        channel_names = []
        for cfg in self.channels:
            channel_names.append(cfg.name)
            if cfg.quality_name is not None:
                channel_names.append(cfg.quality_name)
        channel_names = list(dict.fromkeys(channel_names))
        normalizers = {c.name: copy.deepcopy(c.normalizer) for c in self.channels if c.normalizer is not None}

        classes = set()
        extra_classes = set()
        data_df = edf_to_df(edf_path, channel_names, start=None, end=None, frequency=self.sample_frequency, how=self.resample_type, verbose=True)
        if data_df is None or len(data_df) == 0:
            raise ValueError("Found empty EDF file")

        meta = read_edf_meta(edf_path)
        start = meta["start"]
        end = meta["end"]

        start = max(data_df.index[0], start)
        end = min(data_df.index[-1], end)

        for col in normalizers.keys():
            if col in data_df.columns:
                X = data_df[col].to_numpy(dtype=float).reshape(-1, 1)
                normalizers[col].fit(X=X)

        if self.event_mapping is not None:
            if self.has_extra_target():
                df = self.get_event_df(edf_path, start)
                df_additional = self.get_extra_event_df(edf_path, start)

                if self.remove_unmapped_events:
                    df_additional["Label"] = df_additional["Label"].apply(
                        lambda x: self.event_mapping[x] if x in self.event_mapping else None
                    )
                else:
                    df_additional["Label"] = df_additional["Label"].apply(
                        lambda x: self.event_mapping[x] if x in self.event_mapping else x
                    )
                df_additional = df_additional.dropna()
                extra_classes = set(df_additional["Label"].unique())
            else:
                df = self.get_event_df(edf_path, start)
                df_additional = None

            if self.remove_unmapped_events:
                df["Label"] = df["Label"].apply(lambda x: self.event_mapping[x] if x in self.event_mapping else None)
            else:
                df["Label"] = df["Label"].apply(lambda x: self.event_mapping[x] if x in self.event_mapping else x)
            df = df.dropna()

            if self.force_one_day and (len(df["Starttime"].dt.date.unique()) > 2 or len(df["Endtime"].dt.date.unique()) > 2):
                raise ValueError(
                    f"Edf file: {edf_path} appears to be longer than one entire day. Is this a loading error? If not, set force_one_day = False"
                )

            if self.prepare_patient_callback is not None:
                prepared = self.prepare_patient_callback(
                    data_df=data_df,
                    label_df=df,
                    label_extra_df=df_additional,
                    patient=edf_path,
                )
                if prepared is None:
                    return None
                data_df, df, df_additional = prepared
                if df is None or len(df) == 0:
                    raise ValueError(f"Edf file: {edf_path} was filtered out in prepare_patient")

            classes = set(df["Label"].unique())

            start = max(start, df["Starttime"].min())
            end = min(end, df["Endtime"].max())
        else:
            df = None
            df_additional = None

        return {
            "path": edf_path,
            "data_df": data_df,
            "start": start,
            "end": end,
            "label_df": df,
            "label_extra_df": df_additional,
            "classes": classes,
            "extra_classes": extra_classes,
            "normalizers": normalizers,
        }

    def prepare_patient(self, edf_path) -> Optional[EDFFile]:
        """Prepare one patient recording for lazy window sampling.

        Args:
            edf_path: Path to an EDF file.

        Returns:
            An :class:`EDFFile` descriptor with fitted normalizers, event
            indices, and a precomputed window count, or ``None`` if patient
            preparation rejects the file.

        Raises:
            ValueError: If the file cannot produce at least one valid window or
                appears inconsistent with the configured assumptions.
        """
        try:
            artifacts = self._prepare_patient_artifacts(edf_path)
            if artifacts is None:
                return None

            label_df = artifacts["label_df"]
            label_extra_df = artifacts["label_extra_df"]
            start_offsets = None
            if label_df is not None and len(label_df) > 0:
                start_offsets = self._build_start_offsets(
                    base_start=artifacts["start"],
                    label_df=label_df,
                )
                n_items = len(start_offsets)
            else:
                n_items = int((artifacts["end"] - self.total_input - artifacts["start"]) / self.stride)

            if n_items <= 0:
                raise ValueError(
                    f"Edf file: {edf_path} appears to be empty between {artifacts['start']} - {artifacts['end']} "
                    f"with a total signal length of {artifacts['end'] - artifacts['start']}s"
                )

            return EDFFile(
                path=edf_path,
                X=None,
                channels=list(artifacts["data_df"].columns),
                start_offsets=start_offsets,
                length=n_items,
                labels=EventIndex(label_df) if label_df is not None else None,
                labels_extra=EventIndex(label_extra_df) if label_extra_df is not None else None,
                start_date=artifacts["start"],
                classes=artifacts["classes"].union(artifacts["extra_classes"]),
                normalizers=artifacts["normalizers"],
            )
        except Exception as e:
            logger.warning(f"Cannot read edf file: {edf_path} due to {e}")

            return None #EDFFile(path=edf_path, classes=classes.union(extra_classes))

    def get_patient_stats(self, patients: Sequence[str | os.PathLike], summarize_patient: Callable, num_workers: int = 4) -> pd.DataFrame:
        """Compute per-patient summary rows before dataset initialization.

        Args:
            patients: Patient EDF paths to inspect.
            summarize_patient: Callback receiving ``patient``, ``data_df``,
                ``label_df``, and ``label_extra_df``. It should return a row
                dictionary or ``None``.
            num_workers: Worker count used for parallel summarization.

        Returns:
            A DataFrame built from the rows returned by ``summarize_patient``.

        Notes:
            This helper is intended for cohort-level filtering before
            :meth:`initialize`.
        """
        worker_count = num_workers
        rows: list[dict] = []
        patient_list = list(patients)

        logger.progress_start(len(patient_list), desc="Collecting patient stats", leave=True)
        if worker_count > 1:
            with multiprocessing.Pool(worker_count) as pool:
                iter_objects = pool.imap_unordered(partial(self._summarize_patient, summarize_patient=summarize_patient), patient_list)
                for result in iter_objects:
                    try:
                        if result is not None:
                            rows.append(result)
                    finally:
                        logger.progress_advance(1)
        else:
            for patient in patient_list:
                try:
                    result = self._summarize_patient(patient, summarize_patient)
                    if result is not None:
                        rows.append(result)
                finally:
                    logger.progress_advance(1)

        logger.progress_close()
        logger.info(f"Collected patient stats for {len(rows)}/{len(patient_list)} patients.")
        return pd.DataFrame(rows)

    def _summarize_patient(self, edf_path, summarize_patient: Callable):
        try:
            artifacts = self._prepare_patient_artifacts(edf_path)
            if artifacts is None:
                return None
            row = summarize_patient(
                patient=edf_path,
                data_df=artifacts["data_df"],
                label_df=artifacts["label_df"],
                label_extra_df=artifacts["label_extra_df"],
            )
            if row is None:
                return None
            result = dict(row)
            result.setdefault("patient", edf_path)
            return result
        except Exception as exc:
            logger.warning(f"Failed to summarize patient {edf_path}: {exc}")
            return None

    def initialize(self, patients: Sequence[str | os.PathLike], num_workers: int = 4) -> None:
        """Prepare patient metadata and build the sliding-window index.

        Args:
            patients: EDF paths to include in the dataset.
            num_workers: Worker count used while calling
                :meth:`prepare_patient`.

        Notes:
            Initialization is intentionally explicit because it can be
            expensive. Training scripts often call :meth:`get_patient_stats`
            first to filter patients before paying the full initialization
            cost.
        """
        self.all_patients = list(patients)
        self.initialized = False
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
            self.label_classes = list(self.classes)

        logger.info(
            f"Dataset initialized with {len(self.edf_files)}/{total_n_patients} patients. "
            f"Total windows: {len(self)}. Classes: {len(self.classes)}"
        )
        self.initialized = True

    def run_build_sample(self, item: Dict[str, Any], x_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Finalize one loaded signal window into a training or inference item."""
        quality_df = None
        if len(self.channel_groups) > 0:
            available_columns = list(x_df.columns)
            selected_columns = []
            renamed_columns = list(self.channel_groups.keys())
            selected_quality = {}

            # Emit one sampled representative per configured group and rename the
            # result to the conceptual group name so downstream code sees stable columns.
            for group, group_cfgs in self.channel_configs_by_group.items():
                available = [cfg for cfg in group_cfgs if cfg.name in set(available_columns)]
                if len(available) == 0:
                    raise ValueError(f"No available channels found for group '{group}'.")

                selected_cfg = available[int(np.random.choice(len(available)))]
                selected_columns.append(selected_cfg.name)
                if selected_cfg.quality_name is not None:
                    if selected_cfg.quality_name not in x_df.columns:
                        raise ValueError(
                            f"Missing quality channel '{selected_cfg.quality_name}' for selected channel '{selected_cfg.name}'."
                        )
                    selected_quality[group] = x_df[selected_cfg.quality_name].copy()

            x_selected = x_df.loc[:, selected_columns].copy()
            x_selected.columns = renamed_columns
            x_df = x_selected
            if len(selected_quality) > 0:
                quality_df = pd.DataFrame(selected_quality, index=x_df.index)

        if self.prepare_sample_callback is not None:
            return self.prepare_sample_callback(
                data=x_df,
                quality_data=quality_df,
                target=item.get("target"),
                target_extra=item.get("target_extra"),
                patient=item.get("patient"),
                time=item.get("time"),
                **{k: v for k, v in item.items() if k not in {"data", "target", "target_extra", "patient", "time"}},
            )
        else:
            item["data"] = torch.from_numpy(x_df.values).float()
            return item

    def get_item(self, file: EDFFile, start_date: pd.Timestamp):
        """Build one candidate item from a prepared patient and start time.

        Args:
            file: Prepared patient descriptor.
            start_date: Signal-window start timestamp.

        Returns:
            A sample dictionary or ``None`` when ``prepare_target`` or
            ``prepare_sample`` rejects the candidate.

        Notes:
            Label filtering happens before signal loading when possible, as
            confirmed by ``tests/test_datasets.py``.
        """
        end_date = start_date + self.total_input
        t_center = start_date + (self.total_input // 2 - self.target_resolution // 2)
        item: Dict[str, Any] = {"patient": file.path, "time": t_center}

        if file.labels:
            start_date_label = t_center
            end_date_label = start_date_label + self.target_resolution

            item["target"] = file.get_y(start_date_label, end_date_label, self.sample_frequency, self.label_classes) 
            if file.labels_extra:
                item["target_extra"] = file.get_y_extra(start_date_label, end_date_label, self.sample_frequency, self.label_classes) 

        if self.prepare_target_callback is not None:
            prepared_target = self.prepare_target_callback(
                target=item.get("target"),
                target_extra=item.get("target_extra"),
                patient=item.get("patient"),
                time=item.get("time"),
            )
            if prepared_target is None:
                return None
            if not isinstance(prepared_target, dict):
                raise ValueError(f"prepare_target must return dict or None, but received {type(prepared_target)}.")
            item.update(prepared_target)

        x_df = file.get_x(start_date, end_date, self.sample_frequency, self.resample_type)
        
        if self.rereference:
            for refchannels in self.rereference:
                ref_cols = [r for r in refchannels if r in x_df.columns]
                if ref_cols:
                    x_df[ref_cols] = x_df[ref_cols].values - x_df[ref_cols].values.mean(axis=1)[:,None]

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

        transformed_item = self.run_build_sample(item, x_df)
        if transformed_item is None:
            return None
        item.update(transformed_item)

        return item

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one sample, retrying alternate windows when necessary.

        Args:
            idx: Global window index into the prepared patient list.

        Returns:
            A sample dictionary produced by :meth:`get_item`.

        Raises:
            ValueError: If the dataset was not initialized or if repeated
                rejection exceeds ``online_max_tries``.
        """
        if not self.initialized:
            raise ValueError(f"{self.__class__.__name__} is not initialized. Call initialize(...) before using __getitem__.")
        cnt = 0
        item = None
        last_exception = None
        while True:
            pidx = bisect.bisect_right(self.upper_bounds, idx)
            file = self.edf_files[pidx]

            new_idx = idx - self.lower_bounds[pidx] 
            if file.start_offsets is not None:
                cur_date = file.start_date + self.stride * int(file.start_offsets[new_idx])
            else:
                cur_date = file.start_date + self.stride * new_idx 
            
            try:
                item = self.get_item(file, cur_date)
            except Exception as e:
                last_exception = e
            finally:
                if cnt > self.online_max_tries or item is not None:
                    break
                
                cnt += 1
                idx = int(np.random.randint(self.lower_bounds[pidx], self.upper_bounds[pidx]))
        
        if self.online_max_tries == 0 or cnt <= self.online_max_tries:
            return item
        else:
            raise ValueError(f"Tried to get a clean item for {self.online_max_tries} tries in {self.__class__.__name__ } with no success. Last patient was {file.path}. Exception was {last_exception}")
