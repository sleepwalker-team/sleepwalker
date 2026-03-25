import os
from pathlib import Path
import pytest
from unittest.mock import Mock

import numpy as np
import pandas as pd
import torch

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

from sleepwalker.datasets.ZarrDataset import ZarrDataset, get_zarr_files_in_repo
from sleepwalker.datasets.utils import RepeatSampler, get_edf_files_in_repo
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer

from dotenv import load_dotenv

from tests.utils import iterate_dataset
load_dotenv()  

# TODO: READ NUM_PATIENTS FROM ENVIRONMENT VARIABLE

def build_dataset(dataset_clazz,channel_name, edf_path, num_patients = 5, ending="edf", event_mapping = {}):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True, ending=ending)
    assert len(edf_files) > 0
    edf_files = edf_files[:num_patients]

    dataset = dataset_clazz(patients = edf_files, channels = [ChannelConfig(name=channel_name, normalizer=None)], sample_frequency=100, event_mapping=event_mapping, remove_unmapped_events=False)

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
    task_config = {
        "task": {
            "task": "task",
            "labels": ["n1", "wake"],
            "default": None,
            "percentage": 0.5,
            "target_resolution": pd.Timedelta(seconds=30),
            "n_steps": 1,
        }
    }
    dataset = DummyDataset(
        channels=[ChannelConfig(name="EEG")],
        sample_frequency=1,
        total_input="30s",
        target_resolution="30s",
        event_mapping={"wake": "wake"},
        prepare_target=lambda target, target_extra=None, **_kwargs: MultiLabelTrainer.get_target(
            target=target,
            target_extra=target_extra,
            class_cnts=[1.0, 100.0],
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
    task_config = {
        "task": {
            "task": "task",
            "labels": ["wake"],
            "default": None,
            "percentage": 0.5,
            "target_resolution": pd.Timedelta(seconds=30),
            "n_steps": 1,
        }
    }
    dataset = DummyDataset(
        channels=[ChannelConfig(name="EEG")],
        sample_frequency=1,
        total_input="30s",
        target_resolution="30s",
        event_mapping={"wake": "wake"},
        prepare_target=lambda target, target_extra=None, **_kwargs: MultiLabelTrainer.get_target(
            target=target,
            target_extra=target_extra,
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
    assert item["target"].shape == (1, 1)


def test_grouped_channel_selection_returns_one_channel_per_group():
    dataset = DummyDataset(
        channels=[
            ChannelConfig(name="C3-A2", group="eeg"),
            ChannelConfig(name="C4-A1", group="eeg"),
        ],
        sample_frequency=1,
        total_input="30s",
        target_resolution="30s",
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

    selected = dataset._select_grouped_channels(signal)

    assert list(selected.columns) == ["eeg"]
    assert selected.shape[1] == 1


def test_repeat_sampler_repeats_indices():
    class FixedSampler:
        def __iter__(self):
            return iter([0, 2, 4])

        def __len__(self):
            return 3

    sampler = RepeatSampler(FixedSampler(), n_repeat=3)

    assert list(iter(sampler)) == [0, 0, 0, 2, 2, 2, 4, 4, 4]


def test_grouped_multiclass_trainer_averages_repeats():
    class IdentityModel(torch.nn.Module):
        preprocessors = []

        def forward(self, x):
            return x

    class Loader(list):
        batch_size = 4
        sampler = RepeatSampler([0, 1], n_repeat=2)

    batch = {
        "data": torch.tensor(
            [
                [4.0, 0.0],
                [-2.0, 0.0],
                [0.0, 4.0],
                [0.0, -2.0],
            ]
        ),
        "target": torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
    }
    loader = Loader([batch])
    trainer = MulticlassTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        classes=["wake", "rem"],
        loss_function=torch.nn.functional.cross_entropy,
        device="cpu",
        n_repeat_test=2,
    )
    trainer.steps = {"train": 0, "val": 0, "test": 0}
    trainer.epoch_step = 0

    loss, cm = trainer.run_epoch(loader, None, IdentityModel(), prefix="TEST")

    assert loss >= 0
    assert cm.sum() == 2
    assert cm.trace() == 2


def test_grouped_multiclass_trainer_test_wraps_loader_with_repeat_sampler():
    class IdentityModel(torch.nn.Module):
        preprocessors = []

        def forward(self, x):
            return x

    class FixedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            data = torch.tensor([4.0, 0.0]) if idx == 0 else torch.tensor([0.0, 4.0])
            target = torch.tensor([1.0, 0.0]) if idx == 0 else torch.tensor([0.0, 1.0])
            return {"data": data, "target": target, "patient": str(idx), "time": pd.Timestamp("2024-01-01")}

    loader = torch.utils.data.DataLoader(
        FixedDataset(),
        batch_size=2,
        shuffle=False,
        collate_fn=lambda x: batch_collate(x, ignore_list=["time", "patient"]),
    )
    trainer = MulticlassTrainer(
        epochs=1,
        optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        classes=["wake", "rem"],
        loss_function=torch.nn.functional.cross_entropy,
        device="cpu",
        n_repeat_test=2,
    )
    trainer.steps = {"train": 0, "val": 0, "test": 0}
    trainer.epoch_step = 0

    loss, cm = trainer.test(IdentityModel(), loader)

    assert loss >= 0
    assert cm.sum() == 2
    assert cm.trace() == 2

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

def test_zarr_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    ZARR_PATH = os.environ.get("ZARR_PATH")
    ZARR_PATH = "/raid/sleepwalker/zarr"

    if not ZARR_PATH or not Path(ZARR_PATH).exists():
        pytest.skip("Zarr dataset not available. ")
    
    zarr_files = get_zarr_files_in_repo(ZARR_PATH)

    dataset = ZarrDataset(zarr_files, total_input = "630s", target_resolution = "30s", sample_frequency = 100)
    iterate_dataset(dataset, NUM_BATCHES)

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
    test_zarr_dataset()
