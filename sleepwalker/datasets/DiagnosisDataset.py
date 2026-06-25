# Documentation
from __future__ import annotations

from abc import abstractmethod
import copy
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from mne import event

import numpy as np
import pandas as pd
import torch

from sleepwalker.utils import logger
from sleepwalker.core.signal import edf_to_df, read_edf_meta
from sleepwalker.datasets.Basedataset import BaseDataset, ChannelConfig, EDFFile
from sleepwalker.datasets.normalizer import Normalizer


class DiagnosisDataset(BaseDataset):
    """Patient-level dataset class for diagnosis classification from EDF files.

    Inherits from BaseDataset but overrides sequence sampling and event handling to:
    - Load one label (the diagnosis) per patient file
    - Sample and stack multiple sequences per patient
    """
    def __init__(
            self,
            *,
            channels: Sequence[ChannelConfig],
            sample_frequency: float,
            resample_type: str = "nearest",
            total_input: str | pd.Timedelta = "30s",
            stride: str | pd.Timedelta = None,
            sequences_per_patient: int = 1,
            diagnosis_label_mapping: Optional[Mapping[str, str]] = None,
            remove_unmapped_labels: bool = True,
            prepare_patient: Optional[Callable] = None,
            prepare_target: Optional[Callable] = None,
            prepare_sample: Optional[Callable] = None,
            online_max_tries: int = 128,
            force_one_day: bool = True,
            rereference: Optional[List[List[str]]] = None,
            group_sampling_strategy: Optional[str] = 'random' # NOTE: include?
    ) -> None:
        super().__init__(
            channels=channels,
            sample_frequency=sample_frequency,
            resample_type=resample_type,
            total_input=total_input,
            target_resolution=total_input,
            stride=stride,
            prepare_patient=prepare_patient,
            prepare_target=prepare_target,
            prepare_sample=prepare_sample,
            online_max_tries=online_max_tries,
            force_one_day=force_one_day,
            rereference=rereference,
            group_sampling_strategy=group_sampling_strategy
        )
        self.sequences_per_patient = sequences_per_patient
        self.remove_unmapped_labels = remove_unmapped_labels
        if diagnosis_label_mapping is not None:
            self.diagnosis_label_mapping = {k: v for k, v in diagnosis_label_mapping.items()}
            self.classes = sorted(list(set(self.diagnosis_label_mapping.values())))
            self.label_classes = list(self.classes)
        else:
            self.diagnosis_label_mapping = None
            self.classes = []
            self.label_classes = []

    @abstractmethod
    def get_patient_label(self, edf_path: str) -> Optional[str]: # NOTE: optional?
        """Extract the diagnosis label for a given patient file.
        Args:
            edf_path: Path to the patient EDF file."
            
        Returns:
            The diagnosis label as a string, or None if not available.
        """
        ...

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        """Signal that diagnosis datasets do not provide event annotations."""
        raise ValueError("DiagnosisDataset does not provide event-level labels.")
    
    def __len__(self) -> int:
        """Return the number of patients in the dataset."""
        return len(self.edf_files)
    
    def _prepare_patient_artifacts(self, edf_path):
        channel_names = []
        for cfg in self.channels:
            channel_names.append(cfg.name)
            if cfg.quality_name is not None:
                channel_names.append(cfg.quality_name)
        channel_names = list(dict.fromkeys(channel_names))
        normalizers = {c.name: copy.deepcopy(c.normalizer) for c in self.channels if c.normalizer is not None}

        classes = set()
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

        if self.diagnosis_label_mapping is not None:
            diagnosis = self.get_patient_label(edf_path)

            if self.remove_unmapped_labels:
                label = self.diagnosis_label_mapping[diagnosis] if diagnosis in self.diagnosis_label_mapping else None
            else:
                label = self.diagnosis_label_mapping[diagnosis] if diagnosis in self.diagnosis_label_mapping else diagnosis
            if label is None:
                raise ValueError(f"Found invalid diagnosis label: {diagnosis} for edf file: {edf_path}.")

            # TODO
            # if self.force_one_day and (len(data_df["Starttime"].dt.date.unique()) > 2 or len(data_df["Endtime"].dt.date.unique()) > 2):
            #     raise ValueError(
            #         f"Edf file: {edf_path} appears to be longer than one entire day. Is this a loading error? If not, set force_one_day = False"
            #     )

            if self.prepare_patient_callback is not None:
                prepared = self.prepare_patient_callback(
                    data_df=data_df,
                    label=label, #NOTE: call label or diagnosis?
                    patient=edf_path,
                )
                if prepared is None:
                    return None
                data_df, df, df_additional = prepared
                if label is None or len(data_df) == 0:
                    raise ValueError(f"Edf file: {edf_path} was filtered out in prepare_patient")

            classes = label

            # start = max(start, df["Starttime"].min())
            # end = min(end, df["Endtime"].max())
        else:
            label = None

        return {
            "path": edf_path,
            "data_df": data_df,
            "start": start,
            "end": end,
            "label": label,
            "classes": classes,
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

            start_offsets = None
            # if label_df is not None and len(label_df) > 0:
            #     start_offsets = self._build_start_offsets(
            #         base_start=artifacts["start"],
            #         label_df=label_df,
            #     )
            #     n_items = len(start_offsets)
            # else:
            #     n_items = int((artifacts["end"] - self.total_input - artifacts["start"]) / self.stride)
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
                start_offsets=start_offsets, # NOTE: what are start_offsets?
                length=n_items,
                diagnosis_label=artifacts["label"],
                start_date=artifacts["start"],
                # classes=artifacts["classes"].union(artifacts["extra_classes"]),
                normalizers=artifacts["normalizers"],
            )
        except Exception as e:
            logger.warning(f"Cannot read edf file: {edf_path} due to {e}")

            return None #EDFFile(path=edf_path, classes=classes.union(extra_classes))
        
    def _summarize_patient(self, edf_path, summarize_patient):
        raise NotImplementedError("Summarization not yet implemented for DiagnosisDataset.")
    
    def run_build_sample(self, item: Dict[str, Any], x_dfs: List[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Finalize a list of loaded signal windows into a single training or inference item with stacked data sequences."""
        quality_segments = None
        if len(self.channel_groups) > 0:
            # NOTE: assumes columns are the same for every df in the list
            available_columns = list(x_dfs[0].columns)
            selected_columns = []
            if self.group_sampling_strategy == 'random':
                renamed_columns = list(self.channel_groups.keys())
            elif self.group_sampling_strategy == 'none':
                renamed_columns = sum([[f'{k}' for _ in range(len([channel for channel in v if channel in available_columns]))] for k, v in self.channel_groups.items()], [])
            else:
                raise NotImplementedError('Cannot rename columns for group_sampling_strategy', self.group_sampling_strategy)
            selected_quality = {}

            # Emit one sampled representative per configured group and rename the
            # result to the conceptual group name so downstream code sees stable columns.
            for group, group_cfgs in self.channel_configs_by_group.items():
                available = [cfg for cfg in group_cfgs if cfg.name in set(available_columns)]
                if len(available) == 0:
                    raise ValueError(f"No available channels found for group '{group}'.")

                if self.group_sampling_strategy == 'random':
                    selected_cfg = available[int(np.random.choice(len(available)))]
                    selected_columns.append(selected_cfg.name)
                    if selected_cfg.quality_name is not None:
                        if selected_cfg.quality_name not in x_dfs[0].columns:
                            raise ValueError(
                                f"Missing quality channel '{selected_cfg.quality_name}' for selected channel '{selected_cfg.name}'."
                            )
                        # Get quality channel for sequence
                        selected_quality[group] = [x_df[selected_cfg.quality_name].copy() for x_df in x_dfs]
                elif self.group_sampling_strategy == 'none':
                    # TODO: Quality not supported right now
                    selected_columns.extend([c.name for c in available])
                else:
                    raise NotImplementedError('Cannot sample for group_sampling_strategy', self.group_sampling_strategy)

            # Apply channel selection to each segment and stack
            for i in range(len(x_dfs)):
                x_selected = x_dfs[i].loc[:, selected_columns].copy()
                x_selected.columns = renamed_columns
                x_dfs[i] = x_selected

            if len(selected_quality) > 0:
                quality_segments = []
                for i in range(len(x_dfs)):
                    q_df = pd.DataFrame({g: selected_quality[g][i] for g in selected_quality})
                    quality_segments.append(q_df)

        if self.prepare_sample_callback is not None:
            return self.prepare_sample_callback(
                data=item["data"],
                quality_data=item.get("quality_data"),
                target=item.get("target"),
                patient=item.get("patient"),
                time=item.get("time"),
                **{k: v for k, v in item.items() if k not in {"data", "target", "patient", "time"}},
            )
        else:
            # Stack segments
            item["data"] = torch.stack([torch.from_numpy(df.values).float() for df in x_dfs])
            if quality_segments is not None:
                item["quality_data"] = torch.stack([torch.from_numpy(df.values).float() for df in quality_segments])
        return item
    
    def get_item(self, file: EDFFile, start_dates: list[pd.Timestamp]):
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
        item: Dict[str, Any] = {"patient": file.path, "time": start_dates}

        if file.diagnosis_label:
            item["target"] = file.diagnosis_label

        if self.prepare_target_callback is not None:
            prepared_target = self.prepare_target_callback(
                target=item.get("target"),
            )
            if prepared_target is None:
                return None
            if not isinstance(prepared_target, dict):
                raise ValueError(f"prepare_target must return dict or None, but received {type(prepared_target)}.")
            item.update(prepared_target)

        segments = []
        for start_date in start_dates:
            end_date = start_date + self.total_input
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

            segments.append(x_df)

        transformed_item = self.run_build_sample(item, segments)
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
            file = self.edf_files[idx]

            m = self.sequences_per_patient
            T = file.length

            # TODO: Rework sampling start_indices
            start_indices = np.random.randint(0, T, size=m)
            starts = [file.start_date + self.stride * i for i in start_indices]
            
            try:
                item = self.get_item(file, starts)
            except Exception as e:
                last_exception = e
            finally:
                if cnt > self.online_max_tries or item is not None:
                    break
                
                cnt += 1
                # NOTE: Sample new file idx or not?
                idx = np.random.randint(0, len(self.edf_files))
        if self.online_max_tries == 0 or cnt <= self.online_max_tries:
            return item
        else:
            raise ValueError(f"Tried to get a clean item for {self.online_max_tries} tries in {self.__class__.__name__ } with no success. Last patient was {file.path}. Exception was {last_exception}")

