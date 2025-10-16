from functools import partial
import os
from pathlib import Path
import pytest
from torch.utils.data import DataLoader
import pandas as pd

from sleepwalker.datasets.Basedataset import ChannelConfig, batch_collate
from sleepwalker.datasets.CAP import CAP
from sleepwalker.datasets.MNC import MNC
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

from sleepwalker.datasets.utils import get_edf_files_in_repo

from dotenv import load_dotenv
load_dotenv()  

def run_test(dataset_clazz, channel_name, edf_path, num_batches, batch_size = 8, num_patients = 5, ending="edf"):
    edf_files = get_edf_files_in_repo(edf_path, recursive=True, ending=ending)
    assert len(edf_files) > 0
    edf_files = edf_files[:num_patients]

    dataset = dataset_clazz(patients = edf_files, channels = [ChannelConfig(name=channel_name, normalizer=None)], sample_frequency=100, event_mapping={}, remove_unmapped_events=False)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn= lambda x: batch_collate(x, ignore_list=["time", "patient", "target", "target_extra"]))

    cnt = 0
    assert len(loader) > 0

    for batch in loader:
        assert "patient" in batch
        assert "time" in batch
        assert "data" in batch
        assert "target" in batch

        # check events can be loaded
        edf_path = batch["patient"][0]
        start_dt = pd.Timestamp(batch["time"][0])
        df = dataset.get_event_df(edf_path, start_dt)
        assert not df.empty

        if dataset.has_extra_target():
            assert "target_extra" in batch

            extra_df = dataset.get_extra_event_df(edf_path, start_dt)
            assert not extra_df.empty

        cnt += 1
        if cnt >= num_batches:
            break

def test_synthetic_dataset():
    NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 5))
    edf_data_dir = os.path.join(Path(__file__).parent, "..", "data")

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

    if not EDF_PATH or not Path(EDF_PATH).exists():
        pytest.skip("SVUH_UCD dataset not available")

    run_test(SVUH_UCD, "EMG", EDF_PATH, NUM_BATCHES, ending="rec")

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

if __name__ == '__main__':
    test_isruc_dataset()
    test_ruhrlandklinik_dataset()
    test_sleepedfx_dataset()
    test_cap_dataset()
    test_synthetic_dataset()
    test_shhs_dataset()
    test_svuh_ucd_dataset()
    test_abc_dataset()
    test_apples_dataset() 
    test_hchs_dataset()
    test_mnc_dataset()
    test_mros_dataset()
    test_nchsdb_dataset() 
    test_numom2b_dataset() 
    test_stages_dataset() 
    test_wsc_dataset() 
