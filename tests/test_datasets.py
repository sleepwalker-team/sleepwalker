import os
from pathlib import Path
import pytest

from sleepwalker.datasets.Basedataset import ChannelConfig
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
from sleepwalker.datasets.utils import get_edf_files_in_repo

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