import os
from pathlib import Path
import random
import pytest
from unittest.mock import Mock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sleepwalker.datasets.Basedataset import ChannelConfig, BaseDataset, EDFFile, EventIndex, batch_collate
from sleepwalker.datasets.CAP import CAP
from sleepwalker.datasets.MNC import MNC
from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.NCHSDB import NCHSDB
from sleepwalker.datasets.SHHS import SHHS
from sleepwalker.datasets.ABC import ABC
from sleepwalker.datasets.ISRUC import ISRUC
from sleepwalker.datasets.Ruhrlandklinik import Ruhrlandklinik
from sleepwalker.datasets.SleepEDFx import SleepEDFx
from sleepwalker.datasets.SVUH_UCD import SVUH_UCD
from sleepwalker.datasets.Apples import Apples
from sleepwalker.datasets.HCHS import HCHS
from sleepwalker.datasets.MROS import MROS
from sleepwalker.datasets.Numom2b import Numom2b
from sleepwalker.datasets.WSC import WSC
from sleepwalker.datasets.Stages import Stages
from sleepwalker.datasets.SyntheticDataset import SyntheticDataset

from sleepwalker.datasets.NumpyDataset import NumpyDataset
from sleepwalker.datasets.utils import export_batch_collate, export_dataloader_to_numpy_dir, get_edf_files_in_repo
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.utils.targets import normalize_multitask_config, prepare_multitask_target, slice_target_interval
from sleepwalker.training.execution import RepeatedViewModel
from sleepwalker.training.loader import build_loader
from dotenv import load_dotenv

from tests.utils import iterate_dataset
load_dotenv()  

# TODO: READ NUM_PATIENTS FROM ENVIRONMENT VARIABLE

def build_dataset(dataset_clazz,channel_name, edf_path, num_patients = 5, ending="edf", event_mapping = {}):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True, ending=ending)
    assert len(edf_files) > 0
    edf_files = edf_files[:num_patients]

    dataset = dataset_clazz(channels=[ChannelConfig(channel_name, [channel_name])], sample_frequency=100, event_mapping=event_mapping, remove_unmapped_events=False)
    dataset.initialize(edf_files)

    return dataset

def run_test(dataset_clazz, channel_name, edf_path, num_batches, batch_size = 128, num_patients = 5, ending=".edf"):
    dataset = build_dataset(dataset_clazz, channel_name, edf_path, num_patients, ending)    
    iterate_dataset(dataset, num_batches, batch_size)


class DummyDataset(BaseDataset):
    def initialize(self, patients, num_workers=4):
        self.edf_files = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.initialized = True

    def get_event_df(self, edf_path: str, start_datetime: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError


class OffsetNormalizer:
    def __init__(self, offset):
        self.offset = float(offset)

    def transform(self, data):
        return data + self.offset


def build_export_loader(dataset, batch_size=128):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=export_batch_collate,
        drop_last=False,
    )


def collate_patient_names(samples):
    return [sample["patient"] for sample in samples]


def create_dummy_file() -> EDFFile:
    start = pd.Timestamp("2024-01-01T00:00:00")
    events = pd.DataFrame(
        {
            "Starttime": [start],
            "Endtime": [start + pd.Timedelta(seconds=30)],
            "Label": ["wake"],
        }
    )
    return EDFFile(
        channels=["EEG"],
        path="dummy.edf",
        start_date=start,
        labels=EventIndex(events),
        length=1,
    )


def test_get_item_rejects_before_loading_signal():
    task_config = normalize_multitask_config({
        "task": {
            "labels": ["n1", "wake"],
            "default": None,
            "percentage": 0.5,
            "target_resolution": pd.Timedelta(seconds=30),
            "sequence_len": 1,
        }
    })
    dataset = DummyDataset(
        channels=[ChannelConfig("EEG", ["EEG"])],
        sample_frequency=1,
        total_input="30s",
        event_mapping={"wake": "wake"},
        prepare_target=lambda target, target_extra=None, patient=None, time=None: prepare_multitask_target(
            target=target,
            target_extra=target_extra,
            patient=patient,
            time=time,
            class_cnts={"task": {"n1": 1.0, "wake": 100.0}},
            task_config=task_config,
        ),
    )
    file = create_dummy_file()
    file.get_x = Mock(side_effect=AssertionError("signal should not be loaded"))
    import random
    original_random = random.random
    random.random = lambda: 0.5
    try:
        item = dataset.get_item(file, file.start_date)
    finally:
        random.random = original_random

    assert item is None
    file.get_x.assert_not_called()


def test_get_item_loads_signal_after_label_precheck():
    task_config = normalize_multitask_config({
        "task": {
            "labels": ["wake"],
            "default": None,
            "percentage": 0.5,
            "target_resolution": pd.Timedelta(seconds=30),
            "sequence_len": 1,
        }
    })
    dataset = DummyDataset(
        channels=[ChannelConfig("EEG", ["EEG"])],
        sample_frequency=1,
        total_input="30s",
        event_mapping={"wake": "wake"},
        prepare_target=lambda target, target_extra=None, patient=None, time=None: prepare_multitask_target(
            target=target,
            target_extra=target_extra,
            patient=patient,
            time=time,
            task_config=task_config,
        ),
    )
    file = create_dummy_file()
    signal = pd.DataFrame(
        {"EEG": np.arange(30, dtype=np.float32)},
        index=pd.date_range(file.start_date, periods=30, freq="1s"),
    )
    file.get_x = Mock(return_value=signal)

    item = dataset.get_item(file, file.start_date)

    file.get_x.assert_called_once()
    assert item is not None
    assert torch.equal(item["data"], torch.from_numpy(signal.values).float())
    assert item["target"].shape == (1, 1, 1)
    assert item["target_mask"].tolist() == [[True]]


def test_get_item_timestamp_is_input_window_start():
    dataset = DummyDataset(
        channels=[ChannelConfig("EEG", ["EEG"])],
        sample_frequency=1,
        total_input="60s",
        event_mapping={"wake": "wake"},
    )
    file = create_dummy_file()
    signal = pd.DataFrame(
        {"EEG": np.arange(60, dtype=np.float32)},
        index=pd.date_range(file.start_date, periods=60, freq="1s"),
    )
    file.get_x = Mock(return_value=signal)
    file.get_y = Mock(return_value=pd.DataFrame({"wake": np.ones(60)}, index=signal.index))

    item = dataset.get_item(file, file.start_date)

    assert item["time"] == file.start_date
    file.get_y.assert_called_once_with(file.start_date, file.start_date + pd.Timedelta("60s"), 1, ["wake"])


def test_target_callback_selects_an_explicit_interval_from_input_annotations():
    start = pd.Timestamp("2024-01-01")
    target = pd.DataFrame({"wake": np.ones(60)}, index=pd.date_range(start, periods=60, freq="1s"))

    selected = slice_target_interval(target, target_resolution="20s", target_offset="30s")

    assert len(selected) == 20
    assert selected.index[0] == start + pd.Timedelta("30s")
    assert selected.index[-1] == start + pd.Timedelta("49s")


def test_target_callback_selects_the_last_interval_from_input_annotations():
    start = pd.Timestamp("2024-01-01")
    target = pd.DataFrame({"wake": np.ones(60)}, index=pd.date_range(start, periods=60, freq="1s"))

    selected = slice_target_interval(target, target_resolution="20s", target_position="last")

    assert len(selected) == 20
    assert selected.index[0] == start + pd.Timedelta("40s")
    assert selected.index[-1] == start + pd.Timedelta("59s")


def test_grouped_channel_selection_returns_one_channel_per_group():
    dataset = DummyDataset(
        channels=[
            ChannelConfig("eeg", ["C3-A2", "C4-A1"]),
        ],
        sample_frequency=1,
        total_input="30s",
        event_mapping={},
        remove_unmapped_events=False,
    )
    signal = pd.DataFrame(
        {
            "C3-A2": np.arange(30, dtype=np.float32),
            "C4-A1": np.arange(30, dtype=np.float32) + 100,
        },
        index=pd.date_range("2024-01-01", periods=30, freq="1s"),
    )

    built = dataset.build_sample_from_window_df({"patient": "p", "time": signal.index[0]}, signal)
    selected = pd.DataFrame(built["data"].numpy(), index=signal.index, columns=["eeg"])

    assert list(selected.columns) == ["eeg"]
    assert selected.shape[1] == 1


def test_get_item_reads_only_selected_physical_aliases_and_quality_channels():
    dataset = DummyDataset(
        channels=[
            ChannelConfig("sleep_eeg", ["C3-M2", "C4-M1"], quality_name={"C3-M2": "C3 quality"}),
            ChannelConfig("arousal_eeg", ["C3-M2", "C4-M1"]),
            ChannelConfig("eog", ["E1-M2", "E2-M1"]),
        ],
        sample_frequency=1,
        total_input="3s",
        event_mapping=None,
        group_sampling_strategy="first",
    )
    start = pd.Timestamp("2024-01-01")
    signal = pd.DataFrame(
        {
            "C3-M2": [1.0, 2.0, 3.0],
            "C4-M1": [4.0, 5.0, 6.0],
            "C3 quality": [0.0, 0.0, 0.0],
            "E1-M2": [7.0, 8.0, 9.0],
            "E2-M1": [10.0, 11.0, 12.0],
        },
        index=pd.date_range(start, periods=3, freq="1s"),
    )
    file = EDFFile(channels=list(signal.columns), path="patient.edf", start_date=start, length=1)
    reads = []

    def get_x(start_date, end_date, sample_frequency, resample_type, channels=None):
        reads.append(list(channels))
        return signal.loc[:, channels].copy()

    file.get_x = get_x
    item = dataset.get_item(file, start)

    assert reads == [["C3-M2", "E1-M2", "C3 quality"]]
    assert item["data"].tolist() == [[1.0, 1.0, 7.0], [2.0, 2.0, 8.0], [3.0, 3.0, 9.0]]


def test_one_physical_channel_can_feed_distinct_logical_preprocessing():
    dataset = DummyDataset(
        channels=[
            ChannelConfig("sleep_eeg", ["C3-M2"], normalizer=OffsetNormalizer(1)),
            ChannelConfig("arousal_eeg", ["C3-M2"], normalizer=OffsetNormalizer(10)),
        ],
        sample_frequency=1,
        total_input="3s",
        event_mapping=None,
    )
    signal = pd.DataFrame({"C3-M2": [0.0, 1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=3, freq="1s"))

    built = dataset.build_sample_from_window_df({"patient": "p", "time": signal.index[0]}, signal)

    assert built["data"].tolist() == [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]]


def test_shared_physical_channel_requires_one_unit_and_no_rereferencing():
    channels = [ChannelConfig("a", ["signal"], unit="uV"), ChannelConfig("b", ["signal"], unit="mV")]
    with pytest.raises(ValueError, match="incompatible target units"):
        DummyDataset(channels=channels, sample_frequency=1, event_mapping=None)

    channels = [ChannelConfig("a", ["signal"]), ChannelConfig("b", ["signal"])]
    with pytest.raises(ValueError, match="rereferencing"):
        DummyDataset(channels=channels, sample_frequency=1, event_mapping=None, rereference=[["signal", "reference"]])


def test_shared_physical_channel_is_z_normalized_once_before_duplication():
    start = pd.Timestamp("2024-01-01")
    dataset = DummyDataset(
        channels=[ChannelConfig("a", ["signal"]), ChannelConfig("b", ["signal"])],
        sample_frequency=1,
        total_input="3s",
        event_mapping=None,
        z_normalize=True,
    )
    signal = pd.DataFrame({"signal": [0.0, 1.0, 2.0]}, index=pd.date_range(start, periods=3, freq="1s"))
    file = EDFFile(channels=["signal"], path="patient.edf", start_date=start, length=1, z_statistics={"signal": (1.0, 1.0)})
    file.get_x = lambda *args, **kwargs: signal.copy()

    item = dataset.get_item(file, start)

    assert item["data"].tolist() == [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]


def test_target_callback_can_discard_an_unusable_extra_target():
    dataset = DummyDataset(
        channels=[ChannelConfig("signal", ["signal"])],
        sample_frequency=1,
        total_input="1s",
        event_mapping=None,
        prepare_target=lambda **kwargs: {"target": torch.tensor([1.0])},
    )
    raw_target = pd.DataFrame({"wake": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1s"))
    file = Mock(
        path="patient.edf",
        labels=True,
        labels_extra=True,
        get_y=Mock(return_value=raw_target),
        get_y_extra=Mock(return_value=raw_target),
    )

    item = dataset.get_target_item(file, pd.Timestamp("2024-01-01"))

    assert torch.equal(item["target"], torch.tensor([1.0]))
    assert "target_extra" not in item


def test_channel_config_resolves_per_physical_normalizers_and_quality_channels():
    normalizer = object()
    config = ChannelConfig(
        logical_name="eeg",
        physical_names=["C3-A2", "C4-A1"],
        normalizer={"C3-A2": normalizer, "C4-A1": None},
        quality_name={"C3-A2": "C3 quality", "C4-A1": "C4 quality"},
        unit="uV",
    )

    assert config.normalizer_for("C3-A2") is normalizer
    assert config.normalizer_for("C4-A1") is None
    assert config.quality_name_for("C3-A2") == "C3 quality"
    assert config.quality_name_for("C4-A1") == "C4 quality"

    with pytest.raises(ValueError, match="unknown physical channels"):
        ChannelConfig("eeg", ["C3-A2"], normalizer={"C4-A1": normalizer})


def test_batch_collate_filters_rejected_samples_and_returns_none_for_empty_batch():
    sample = {"data": torch.ones(2, 1), "patient": "patient", "time": pd.Timestamp("2024-01-01")}

    batch = batch_collate([None, sample])

    assert batch["data"].shape == (1, 2, 1)
    assert batch["patient"] == ["patient"]
    assert batch_collate([None, None]) is None


def test_dataset_initialization_does_not_load_complete_signals(monkeypatch):
    import sleepwalker.datasets.Basedataset as basedataset_module

    dataset = SyntheticDataset(
        channels=[ChannelConfig("EEG", ["EEG"])],
        sample_frequency=100,
        event_mapping={},
        remove_unmapped_events=False,
    )
    edf_path = Path(__file__).parent / "data" / "signals_01.edf"
    monkeypatch.setattr(basedataset_module, "edf_to_df", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("initialize loaded the signal")))

    dataset.initialize([edf_path], num_workers=0, strict=True)

    assert dataset.get_n_patients() == 1
    assert dataset.get_patient_ranges() == [(0, len(dataset))]


def test_recording_z_normalization_runs_after_normalizers_and_rereferencing(monkeypatch):
    import sleepwalker.datasets.Basedataset as basedataset_module

    start = pd.Timestamp("2024-01-01")
    full_signal = pd.DataFrame(
        {"A": [0.0, 2.0, 4.0, 6.0], "B": [0.0, 0.0, 0.0, 0.0]},
        index=pd.date_range(start, periods=4, freq="1s"),
    )
    reads = []

    def read_meta(path):
        return {"start": start, "end": start + pd.Timedelta(seconds=4), "signals": ["A", "B"], "units": {"A": "uV", "B": "uV"}, "source": "pyedflib"}

    def read_signal(path, channels, start, end, frequency, how, verbose):
        reads.append((start, end))
        return full_signal.loc[:, channels].copy()

    warnings = []
    monkeypatch.setattr(basedataset_module, "read_edf_meta", read_meta)
    monkeypatch.setattr(basedataset_module, "edf_to_df", read_signal)
    monkeypatch.setattr(basedataset_module.logger, "warning", warnings.append)

    dataset = SyntheticDataset(
        channels=[ChannelConfig("A", ["A"], normalizer=OffsetNormalizer(10.0)), ChannelConfig("B", ["B"])],
        sample_frequency=1,
        total_input="2s",
        stride="1s",
        event_mapping=None,
        rereference=[["A", "B"]],
        z_normalize=True,
    )
    dataset.initialize(["recording.edf"], num_workers=0, strict=True)

    assert reads == [(None, None)]
    assert warnings == ["z_normalize=True with rereferencing enabled: recording z-normalization is applied after rereferencing."]
    assert dataset.edf_files[0].z_statistics == {
        "A": (6.5, np.std([5.0, 6.0, 7.0, 8.0])),
        "B": (-6.5, np.std([-5.0, -6.0, -7.0, -8.0])),
    }

    item = dataset.get_item(dataset.edf_files[0], start)

    expected = torch.tensor([[-1.3416408, 1.3416408], [-0.4472136, 0.4472136]])
    assert torch.allclose(item["data"], expected)


def test_prepare_patient_callback_is_label_only():
    calls = []

    def prepare_patient(label_df, label_extra_df, patient=None):
        calls.append((label_df, label_extra_df, patient))
        return label_df, label_extra_df

    dataset = SyntheticDataset(
        channels=[ChannelConfig("EEG", ["EEG"])],
        sample_frequency=100,
        event_mapping={},
        remove_unmapped_events=False,
        prepare_patient=prepare_patient,
    )
    edf_path = Path(__file__).parent / "data" / "signals_01.edf"

    dataset.initialize([edf_path], num_workers=0, strict=True)

    assert len(calls) == 1
    assert calls[0][0] is not None
    assert calls[0][1] is None
    assert calls[0][2] == edf_path


def test_basedataset_retries_patient_before_switching_global(monkeypatch):
    class RetryDataset(DummyDataset):
        def __init__(self):
            super().__init__(
                channels=[ChannelConfig("EEG", ["EEG"])],
                sample_frequency=1,
                total_input="30s",
                event_mapping={},
                remove_unmapped_events=False,
                online_max_tries=2,
            )
            self.calls = []

        def get_item(self, file, start_date):
            self.calls.append(file.path)
            if file.path == "p1.edf":
                return None
            return {"data": torch.zeros(30, 1), "target": torch.tensor([1.0]), "patient": file.path, "time": start_date}

    dataset = RetryDataset()
    dataset.edf_files = [
        EDFFile(channels=["EEG"], path="p1.edf", start_date=pd.Timestamp("2024-01-01"), length=1),
        EDFFile(channels=["EEG"], path="p2.edf", start_date=pd.Timestamp("2024-01-01"), length=1),
    ]
    dataset.lower_bounds = [0, 1]
    dataset.upper_bounds = [1, 2]
    dataset.initialized = True

    warnings = []
    monkeypatch.setattr(np.random, "randint", lambda lower, upper: 0)
    monkeypatch.setattr(np.random, "choice", lambda values: 1)
    monkeypatch.setattr("sleepwalker.datasets.Basedataset.logger.warning", warnings.append)

    item = dataset[0]

    assert item["patient"] == "p2.edf"
    assert dataset.calls == ["p1.edf", "p1.edf", "p2.edf"]
    assert warnings == []


def test_basedataset_propagates_item_exceptions():
    class FailingDataset(DummyDataset):
        def __init__(self):
            super().__init__(
                channels=[ChannelConfig("EEG", ["EEG"])],
                sample_frequency=1,
                total_input="30s",
                event_mapping={},
                remove_unmapped_events=False,
                online_max_tries=1,
            )

        def get_item(self, file, start_date):
            raise RuntimeError(f"Cannot load {file.path}")

    dataset = FailingDataset()
    dataset.edf_files = [
        EDFFile(channels=["EEG"], path="p1.edf", start_date=pd.Timestamp("2024-01-01"), length=1),
        EDFFile(channels=["EEG"], path="p2.edf", start_date=pd.Timestamp("2024-01-01"), length=1),
    ]
    dataset.lower_bounds = [0, 1]
    dataset.upper_bounds = [1, 2]
    dataset.initialized = True

    with pytest.raises(RuntimeError, match="Cannot load p1.edf"):
        dataset[0]


def test_basedataset_returns_none_without_resampling():
    class RejectingDataset(DummyDataset):
        def get_item(self, file, start_date):
            return None

    dataset = RejectingDataset(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=1, event_mapping={}, remove_unmapped_events=False, rejection_strategy="none")
    dataset.edf_files = [EDFFile(channels=["EEG"], path="p1.edf", start_date=pd.Timestamp("2024-01-01"), length=1)]
    dataset.lower_bounds = [0]
    dataset.upper_bounds = [1]
    dataset.initialized = True

    assert dataset[0] is None


def test_basedataset_repeats_views_atomically_before_fallback(monkeypatch):
    class RepeatedDataset(DummyDataset):
        def __init__(self):
            super().__init__(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=1, event_mapping={}, remove_unmapped_events=False, online_max_tries=1, n_views=2)
            self.calls = []

        def get_item(self, file, start_date):
            self.calls.append(file.path)
            if file.path == "p1.edf" and self.calls.count(file.path) == 2:
                return None
            return {"data": torch.full((2, 1), float(len(self.calls))), "target": torch.tensor([1.0]), "patient": file.path, "time": start_date}

    dataset = RepeatedDataset()
    dataset.edf_files = [
        EDFFile(channels=["EEG"], path="p1.edf", start_date=pd.Timestamp("2024-01-01"), length=1),
        EDFFile(channels=["EEG"], path="p2.edf", start_date=pd.Timestamp("2024-01-01"), length=1),
    ]
    dataset.lower_bounds = [0, 1]
    dataset.upper_bounds = [1, 2]
    dataset.initialized = True
    monkeypatch.setattr(np.random, "choice", lambda values: 1)

    item = dataset[0]

    assert dataset.calls == ["p1.edf", "p1.edf", "p2.edf", "p2.edf"]
    assert item["patient"] == "p2.edf"
    assert item["data"].shape == (2, 2, 1)


def test_loader_filters_rejections_with_multiple_workers():
    class RejectingDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 6

        def __getitem__(self, index):
            if index % 2 == 0:
                return None
            return {"patient": str(index)}

    loader = build_loader(RejectingDataset(), batch_size=3, num_workers=2, n_samples=None, collate_fn=collate_patient_names, sampling="sequential", seed=17)
    batches = [batch for batch in loader if batch is not None]

    assert [patient for batch in batches for patient in batch] == ["1", "3", "5"]


def test_loader_drop_last_discards_batches_with_rejected_samples():
    class RejectingDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 6

        def __getitem__(self, index):
            if index == 4:
                return None
            return {"patient": str(index)}

    loader = build_loader(RejectingDataset(), batch_size=3, num_workers=0, n_samples=None, collate_fn=collate_patient_names, sampling="sequential", seed=17, drop_last=True)

    assert [batch for batch in loader if batch is not None] == [["0", "1", "2"]]


def test_multidataset_propagates_rejected_samples():
    class RejectingDataset(DummyDataset):
        def get_item(self, file, start_date):
            return None

    dataset = RejectingDataset(channels=[ChannelConfig("EEG", ["EEG"])], sample_frequency=1, event_mapping={}, remove_unmapped_events=False, rejection_strategy="none")
    dataset.edf_files = [EDFFile(channels=["EEG"], path="p1.edf", start_date=pd.Timestamp("2024-01-01"), length=1)]
    dataset.lower_bounds = [0]
    dataset.upper_bounds = [1]
    dataset.initialized = True

    assert MultiDataset([dataset])[0] is None


def test_grouped_multiclass_trainer_averages_repeats():
    class IdentityModel(torch.nn.Module):
        preprocessors = []

        def forward(self, x):
            return x.unsqueeze(1)

    class Loader(list):
        batch_size = 2
        sampler = [0, 1]
        dataset = type("Dataset", (), {})()

    batch = {
        "data": torch.tensor(
            [[[4.0, 0.0], [-2.0, 0.0]], [[0.0, 4.0], [0.0, -2.0]]]
        ),
        "target": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]]
        ).unsqueeze(1),
    }
    loader = Loader([batch])
    trainer = MulticlassTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        classes=["wake", "rem"],
        loss_function=torch.nn.functional.cross_entropy,
        target_resolution="30s",
        device="cpu",
    )
    trainer.steps = {"train": 0, "val": 0, "test": 0}
    trainer.epoch_step = 0

    loss, cm = trainer.run_epoch(loader, None, RepeatedViewModel(IdentityModel()), prefix="TEST")

    assert loss >= 0
    assert cm.sum() == 2
    assert cm.trace() == 2


def test_multiclass_training_allows_an_exhausted_loader():
    class Loader(list):
        batch_size = 2
        dataset = type("Dataset", (), {})()

    model = torch.nn.Linear(1, 1)
    trainer = MulticlassTrainer(
        epochs=1,
        optimizer=lambda current_model: torch.optim.SGD(current_model.parameters(), lr=0.1),
        classes=["wake", "rem"],
        loss_function=torch.nn.functional.cross_entropy,
        target_resolution="30s",
        device="cpu",
    )
    trainer.steps = {"train": 0, "val": 0, "test": 0}
    trainer.epoch_step = 0

    loss, cm = trainer.run_epoch(Loader([None]), torch.optim.SGD(model.parameters(), lr=0.1), model, prefix="TRAIN")

    assert loss == 0
    assert cm.tolist() == [[0, 0], [0, 0]]


def test_forward_batch_repeated_view_average_keeps_training_gradients():
    model = torch.nn.Linear(1, 1, bias=False)
    repeated_model = RepeatedViewModel(model)
    outputs = repeated_model(torch.tensor([[[[1.0]], [[3.0]]], [[[2.0]], [[4.0]]]]))
    outputs.sum().backward()

    assert outputs.shape == (2, 1, 1)
    assert model.weight.grad is not None


def test_multiclass_trainer_rejects_epoch_len_target_step_mismatch():
    class EpochModel(torch.nn.Module):
        epoch_len_s = 1.0

        def forward(self, x):
            return torch.zeros((x.shape[0], 40, 2))

    class Loader(list):
        batch_size = 1
        sampler = torch.utils.data.SequentialSampler([0])
        dataset = type("Dataset", (), {})()

    trainer = MulticlassTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        classes=["no_arousal", "arousal"],
        loss_function=torch.nn.functional.cross_entropy,
        target_resolution="60s",
        device="cpu",
        sequence_len=40,
    )

    with pytest.raises(ValueError, match="epoch_len"):
        trainer.run_epoch(Loader(), None, EpochModel(), prefix="TEST")


def test_multiclass_trainer_excludes_masked_sequence_steps():
    class IdentityModel(torch.nn.Module):
        def forward(self, x):
            return x

    class Loader(list):
        batch_size = 1
        sampler = torch.utils.data.SequentialSampler([0])
        dataset = type("Dataset", (), {})()

    batch = {
        "data": torch.tensor([[[4.0, 0.0], [4.0, 0.0]]]),
        "target": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        "target_mask": torch.tensor([[True, False]]),
    }
    trainer = MulticlassTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        classes=["event", "no event"],
        loss_function=torch.nn.functional.cross_entropy,
        target_resolution="2s",
        device="cpu",
        sequence_len=2,
    )
    trainer.steps = {"train": 0, "val": 0, "test": 0}
    trainer.epoch_step = 0

    loss, cm = trainer.run_epoch(Loader([batch]), None, IdentityModel(), prefix="TEST")

    assert loss >= 0
    assert cm.sum() == 1
    assert cm.trace() == 1


def test_repeat_loader_and_model_wrapper_feed_regular_trainer():
    class IdentityModel(torch.nn.Module):
        preprocessors = []

        def forward(self, x):
            return x.unsqueeze(1)

    class FixedDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.n_views = 1

        def set_n_views(self, n_views):
            self.n_views = n_views

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            data = torch.tensor([4.0, 0.0]) if idx == 0 else torch.tensor([0.0, 4.0])
            if self.n_views > 1:
                data = data.unsqueeze(0).expand(self.n_views, -1).clone()
            target = torch.tensor([[1.0, 0.0]]) if idx == 0 else torch.tensor([[0.0, 1.0]])
            return {"data": data, "target": target, "patient": str(idx), "time": pd.Timestamp("2024-01-01")}

    loader = build_loader(
        FixedDataset(),
        batch_size=2,
        num_workers=0,
        n_samples=None,
        seed=17,
        sampling="sequential",
        collate_fn=lambda x: batch_collate(x, ignore_list=["time", "patient"]),
        n_views=2,
    )
    trainer = MulticlassTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        classes=["wake", "rem"],
        loss_function=torch.nn.functional.cross_entropy,
        target_resolution="30s",
        device="cpu",
    )
    trainer.steps = {"train": 0, "val": 0, "test": 0}
    trainer.epoch_step = 0

    loss, cm = trainer.test(RepeatedViewModel(IdentityModel()), loader)

    assert loss >= 0
    assert cm.sum() == 2
    assert cm.trace() == 2


def test_multiclass_trainer_balance_gamma_changes_acceptance_strength(monkeypatch):
    trainer_default = MulticlassTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        classes=["majority", "minority"],
        loss_function=torch.nn.functional.cross_entropy,
        target_resolution="30s",
        device="cpu",
        balance_batches=True,
        balance_gamma=1.0,
    )
    trainer_stronger = MulticlassTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        classes=["majority", "minority"],
        loss_function=torch.nn.functional.cross_entropy,
        target_resolution="30s",
        device="cpu",
        balance_batches=True,
        balance_gamma=2.0,
    )

    target = torch.tensor([1.0, 0.0])
    class_cnts = [9.0, 1.0]

    monkeypatch.setattr(random, "random", lambda: 0.05)
    assert trainer_default._keep_balanced_target(target, class_cnts) is True
    assert trainer_stronger._keep_balanced_target(target, class_cnts) is False


def test_multiclass_trainer_uses_configured_class_counts_without_loading_data(monkeypatch):
    trainer = MulticlassTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        classes=["majority", "minority"],
        loss_function=torch.nn.functional.cross_entropy,
        target_resolution="30s",
        loss_mode="inverse",
        class_counts={"majority": 9.0, "minority": 1.0},
        device="cpu",
    )
    dataset = type("Dataset", (), {"get_classes": lambda self: ["majority", "minority"]})()
    loader = type("Loader", (), {"dataset": dataset})()
    monkeypatch.setattr("sleepwalker.trainer.MulticlassTrainer.estimate_class_cnts", lambda current_loader: pytest.fail("configured counts must skip online estimation"))

    trainer.warmup_trainer(loader)

    assert trainer.loss_function.keywords["weight"][1] > trainer.loss_function.keywords["weight"][0]

def test_synthetic_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    edf_data_dir = os.path.join(Path(__file__).parent, "data")

    run_test(SyntheticDataset, "EEG", edf_data_dir, NUM_BATCHES)

def test_cap_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("CAP_PATH")

    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("CAP dataset not available")

    run_test(CAP, "Fp2-F4", EDF_PATH, NUM_BATCHES)

def test_ruhrlandklinik_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("RUHRLANDKLINIK_PATH")
    
    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("RUHRLANDKLINIK_PATH dataset not available")

    run_test(Ruhrlandklinik, "C4", EDF_PATH, NUM_BATCHES)

def test_sleepedfx_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("SLEEP_EDFX_PATH")
    
    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("SLEEP_EDFX_PATH dataset not available")

    run_test(SleepEDFx, "EEG Fpz-Cz", EDF_PATH, NUM_BATCHES)

def test_isruc_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("ISRUC_PATH") 

    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("ISRUC dataset not available")

    run_test(ISRUC, "E1-M2", EDF_PATH, NUM_BATCHES)
    
def test_shhs_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("SHHS_PATH") 

    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("SHHS dataset not available")

    run_test(SHHS, "EMG", EDF_PATH, NUM_BATCHES)

def test_mros_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("MROS_PATH") 
    
    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("MROS dataset not available")

    run_test(MROS, "C4", EDF_PATH, NUM_BATCHES)

def test_svuh_ucd_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("SVUH_UCD_PATH") 
    EDF_PATH = "/raid/sleepwalker/svuh-ucd"

    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("SVUH_UCD dataset not available")

    run_test(SVUH_UCD, "EMG", EDF_PATH, NUM_BATCHES, ending=".rec")

def test_abc_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("ABC_PATH") 
    
    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("ABC dataset not available")

    run_test(ABC, "C4", EDF_PATH, NUM_BATCHES)

def test_apples_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("APPLES_PATH") 

    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("Apples dataset not available")

    run_test(Apples, "EMG", EDF_PATH, NUM_BATCHES)

def test_hchs_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("HCHS_PATH") 

    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("ABC dataset not available")

    run_test(HCHS, "A-Snore", EDF_PATH, NUM_BATCHES)

def test_mnc_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("MNC_PATH") 

    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("MNC dataset not available")

    run_test(MNC, "E2", EDF_PATH, NUM_BATCHES, num_patients=10)

def test_nchsdb_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("NCHSDB_PATH") 
    
    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("NCHSDB dataset not available")

    run_test(NCHSDB, "EEG F4-M1", EDF_PATH, NUM_BATCHES)

def test_numom2b_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("NUMOM2B_PATH") 

    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("NUMOM2B dataset not available")

    run_test(Numom2b, "ECG", EDF_PATH, NUM_BATCHES)

def test_stages_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("STAGES_PATH") 
    
    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("STAGES dataset not available")

    run_test(Stages, "C4", EDF_PATH, NUM_BATCHES)

def test_wsc_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    EDF_PATH = os.environ.get("WSC_PATH") 
    
    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("WSC dataset not available")

    run_test(WSC, "ECG", EDF_PATH, NUM_BATCHES)

def test_multi_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    CAP_PATH = os.environ.get("CAP_PATH") 
    if not CAP_PATH or not Path(CAP_PATH).exists():
        pytest.skip("CAP dataset not available. ")

    edf_data_dir = os.path.join(Path(__file__).parent, "data")

    ds1 = build_dataset(
        SyntheticDataset, 
        "EEG", 
        edf_data_dir,
        event_mapping = {
            "n1":"n1",
            "n2":"n2",
            "n3":"n3",
            "rem":"rem",
            "wake":"wake"
        }
    )

    ds2 = build_dataset(
        CAP, 
        "Fp2-F4", 
        CAP_PATH, 
        event_mapping = {
            "s1":"n1",
            "s2":"n2",
            "s3":"n3",
            "s4":"n3",
            "r":"rem",
            "w":"wake"
        }
    )

    ds = MultiDataset([ds1,ds2])
    iterate_dataset(ds, NUM_BATCHES)

if __name__ == '__main__':
    # test_isruc_dataset()
    # test_ruhrlandklinik_dataset()
    # test_sleepedfx_dataset()
    # test_synthetic_dataset()
    # test_shhs_dataset()
    test_svuh_ucd_dataset()
    # test_abc_dataset()
    # test_apples_dataset() 
    # test_hchs_dataset()
    # test_mnc_dataset()
    # test_mros_dataset()
    # test_nchsdb_dataset() 
    # test_numom2b_dataset() 
    # test_stages_dataset() 
    # test_wsc_dataset() 
    # test_multi_dataset()
    # test_cap_dataset()


class CacheReadyDataset(DummyDataset):
    def __init__(self):
        super().__init__(
            channels=[ChannelConfig("eeg", ["eeg"]), ChannelConfig("emg", ["emg"])],
            sample_frequency=2,
            total_input="2s",
            event_mapping=None,
        )
        self.classes = ['wake', 'rem']
        self.label_classes = list(self.classes)
        self.initialized = True
        self.all_patients = ['patient-a', 'patient-b']
        self._files = []
        base_time = pd.Timestamp('2024-01-01')
        for idx in range(3):
            base = idx * 10
            signal = pd.DataFrame(
                [
                    [base + 1, base + 2],
                    [base + 3, base + 4],
                    [base + 5, base + 6],
                    [base + 7, base + 8],
                ],
                columns=["eeg", "emg"],
                index=pd.date_range(base_time + pd.Timedelta(seconds=idx) - pd.Timedelta(milliseconds=500), periods=4, freq="500ms"),
            )
            self._files.append(
                EDFFile(
                    channels=["eeg", "emg"],
                    path='patient-a' if idx < 2 else 'patient-b',
                    start_date=base_time + pd.Timedelta(seconds=idx) - pd.Timedelta(milliseconds=500),
                    length=1,
                    X=signal,
                )
            )

    def __len__(self):
        return 3

    def get_classes(self):
        return list(self.classes)

    def has_extra_target(self):
        return True

    def get_n_patients(self):
        return 2

    def get_timeseries_len(self):
        return 4

    def __getitem__(self, idx):
        base = idx * 10
        return {
            'data': torch.tensor([[base + 1, base + 2], [base + 3, base + 4], [base + 5, base + 6], [base + 7, base + 8]], dtype=torch.float32),
            'target': torch.tensor([1.0, 0.0], dtype=torch.float32) if idx % 2 == 0 else torch.tensor([0.0, 1.0], dtype=torch.float32),
            'target_extra': torch.tensor([float(idx), float(idx + 1)], dtype=torch.float32),
            'patient': 'patient-a' if idx < 2 else 'patient-b',
            'time': pd.Timestamp('2024-01-01') + pd.Timedelta(seconds=idx),
            'dataset': idx % 2,
        }


class GroupedCacheDataset(DummyDataset):
    def __init__(self):
        super().__init__(
            channels=[
                ChannelConfig("eeg", ["channel-a", "channel-b"]),
            ],
            sample_frequency=1,
            total_input="3s",
            event_mapping=None,
        )
        self.classes = ['wake']
        self.label_classes = list(self.classes)
        self.initialized = True
        self.all_patients = ['patient-a']
        self._file = EDFFile(
            channels=["channel-a", "channel-b"],
            path="patient-a",
            start_date=pd.Timestamp("2023-12-31 23:59:59"),
            length=1,
            X=pd.DataFrame(
                {
                    "channel-a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    "channel-b": np.array([11.0, 12.0, 13.0], dtype=np.float32),
                },
                index=pd.date_range("2024-01-01", periods=3, freq="1s"),
            ),
        )
        self.edf_files = [self._file]
        self.lower_bounds = [0]
        self.upper_bounds = [1]

    def __len__(self):
        return 1

    def get_classes(self):
        return list(self.classes)

    def has_extra_target(self):
        return False

    def get_n_patients(self):
        return 1

    def get_timeseries_len(self):
        return 3

    def __getitem__(self, idx):
        return super().__getitem__(idx)


def test_numpy_dataset_export_roundtrip(tmp_path):
    cache_dir = export_dataloader_to_numpy_dir(
        DataLoader(
            CacheReadyDataset(),
            batch_size=128,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: export_batch_collate(batch, extra_keys=["dataset"]),
            drop_last=False,
        ),
        tmp_path / 'cache',
        extra_keys=['dataset'],
    )

    dataset = NumpyDataset(cache_dir)

    assert len(dataset) == 3
    assert dataset.get_classes() == ['wake', 'rem']
    assert dataset.has_extra_target() is True
    assert dataset.get_input_channels() == ['eeg', 'emg']

    item = dataset[1]
    assert torch.equal(item['data'], torch.tensor([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]))
    assert torch.equal(item['target'], torch.tensor([0.0, 1.0]))
    assert torch.equal(item['target_extra'], torch.tensor([1.0, 2.0]))
    assert item['patient'] == 'patient-a'
    assert item['time'] == pd.Timestamp('2024-01-01 00:00:01')
    assert item['dataset'] == 1


def test_numpy_dataset_supports_memmap(tmp_path):
    cache_dir = export_dataloader_to_numpy_dir(build_export_loader(CacheReadyDataset()), tmp_path / 'cache')

    dataset = NumpyDataset(cache_dir, in_memory=False)

    assert len(dataset) == 3
    assert dataset[0]['data'].shape == (4, 2)


def test_numpy_cache_freezes_one_export_pass(tmp_path, monkeypatch):
    source = GroupedCacheDataset()
    choices = iter([1, 0, 0])
    monkeypatch.setattr(np.random, 'choice', lambda values: next(choices))

    cache_dir = export_dataloader_to_numpy_dir(build_export_loader(source), tmp_path / 'cache')
    cached = NumpyDataset(cache_dir)

    first = cached[0]['data'].clone()
    second = cached[0]['data'].clone()
    assert torch.equal(first, second)
    assert torch.equal(first, torch.tensor([[11.0], [12.0], [13.0]]))

    live_item = source[0]['data']
    assert torch.equal(live_item, torch.tensor([[1.0], [2.0], [3.0]]))


def test_numpy_dataset_chunked_roundtrip(tmp_path):
    cache_dir = export_dataloader_to_numpy_dir(
        DataLoader(
            CacheReadyDataset(),
            batch_size=128,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: export_batch_collate(batch, extra_keys=["dataset"]),
            drop_last=False,
        ),
        tmp_path / 'cache_chunked',
        extra_keys=['dataset'],
        n_samples_per_file=2,
    )

    assert (cache_dir / 'data.000000.npy').exists()
    assert (cache_dir / 'data.000001.npy').exists()

    dataset = NumpyDataset(cache_dir, in_memory=False)

    assert len(dataset) == 3
    item = dataset[2]
    assert torch.equal(item['data'], torch.tensor([[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0]]))
    assert torch.equal(item['target'], torch.tensor([1, 0]))
    assert item['patient'] == 'patient-b'
    assert item['dataset'] == 0
