"""Inference-time dataset template without label extraction.

`UnlabelledDataset` reuses the EDF loading and window-building logic from
`BaseDataset` but disables all label handling. It is primarily used by the
prediction-package export/load path and by inference helpers that score raw EDF
files without ground-truth annotations.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional, Sequence

import pandas as pd

from sleepwalker.datasets.Basedataset import BaseDataset, ChannelConfig, prepare_tensor_sample


class UnlabelledDataset(BaseDataset):
    """Dataset template for EDF inference without labels.

    Args:
        channels: Signal channels to load.
        sample_frequency: Resampling frequency in Hz.
        resample_type: Signal resampling mode.
        total_input: Input window duration.
        stride: Time between consecutive inference windows.
        prepare_sample: Optional final sample callback reused from
            `BaseDataset`.
        online_max_tries: Retry budget when `prepare_sample` rejects a window.
        rejection_strategy: Candidate fallback policy after expected rejection.
        n_views: Number of independently prepared views per accepted window.
        prepare_channels: Optional final callback that maps processed source channels to model channels.
        input_channels: Ordered model channel names, required with prepare_channels.
    """
    def __init__(
        self,
        *,
        channels: Sequence[ChannelConfig],
        sample_frequency: float,
        resample_type: str = "nearest",
        total_input: str | pd.Timedelta = "30s",
        stride: str | pd.Timedelta | None = None,
        prepare_sample: Callable = prepare_tensor_sample,
        online_max_tries: int = 128,
        rejection_strategy: str = "patient_then_global",
        n_views: int = 1,
        prepare_channels: Optional[Callable] = None,
        input_channels: Optional[Sequence[str]] = None,
        group_sampling_strategy: str = "first",
    ) -> None:
        self._init_kwargs = {
            "channels": list(channels),
            "sample_frequency": sample_frequency,
            "resample_type": resample_type,
            "total_input": total_input,
            "stride": stride,
            "prepare_sample": prepare_sample,
            "online_max_tries": online_max_tries,
            "rejection_strategy": rejection_strategy,
            "n_views": n_views,
            "prepare_channels": prepare_channels,
            "input_channels": input_channels,
            "group_sampling_strategy": group_sampling_strategy,
        }
        super().__init__(**self._init_kwargs, event_mapping=None, prepare_target=None, force_one_day=False)
        if input_channels is not None:
            self._init_kwargs["input_channels"] = list(self.input_channels)

    @classmethod
    def from_dataset(cls, dataset: BaseDataset) -> UnlabelledDataset:
        """Copy the executable signal pipeline from a configured dataset."""
        if not isinstance(dataset, BaseDataset):
            raise TypeError(f"Cannot derive raw-EDF inference preprocessing from {type(dataset).__name__}.")
        return cls(
            channels=dataset.channels,
            sample_frequency=dataset.sample_frequency,
            resample_type=dataset.resample_type,
            total_input=dataset.total_input,
            stride=dataset.stride,
            prepare_sample=dataset.prepare_sample_callback,
            online_max_tries=dataset.online_max_tries,
            rejection_strategy=dataset.rejection_strategy,
            n_views=dataset.n_views,
            prepare_channels=dataset.prepare_channels_callback,
            input_channels=dataset.get_input_channels() if dataset.prepare_channels_callback is not None else None,
            group_sampling_strategy="first",
        )

    def clone(
        self,
        *,
        channels: Optional[Sequence[ChannelConfig]] = None,
    ) -> UnlabelledDataset:
        """Return a fresh dataset template with the same configuration."""
        kwargs = dict(self._init_kwargs)
        if channels is not None:
            kwargs["channels"] = list(channels)
        return UnlabelledDataset(**kwargs)

    def dataset_kwargs(self) -> dict:
        """Return a fresh copy of the dataset constructor arguments."""
        return copy.deepcopy(self._init_kwargs)

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        """Signal that unlabelled datasets do not provide event annotations."""
        raise ValueError("UnlabelledDataset does not provide labels.")
