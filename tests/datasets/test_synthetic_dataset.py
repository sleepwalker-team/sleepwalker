import os
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader
import pandas as pd

from sleepwalker.datasets.Basedataset import ChannelConfig, batch_collate
from sleepwalker.datasets.SyntheticDataset import SyntheticDataset
from sleepwalker.datasets.utils import get_edf_files_in_repo

def test_synthetic_dataset_with_dataloader():
    edf_data_dir = os.path.join(Path(__file__).parent, "..", "data")

    # collect EDFs from the repo (synthetic data dir)
    edf_files = get_edf_files_in_repo(str(edf_data_dir), recursive=False)
    assert len(edf_files) > 0

    # build dataset (with extra targets enabled)
    ds = SyntheticDataset(patients = edf_files, channels = [ChannelConfig(name="EEG", normalizer=None)], sample_frequency=100, event_mapping={k:k for k in ["wake","rem", "n1", "n2", "n3"]}, has_extra=True)

    # wrap in a DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=batch_collate)

    # iterate through a few batches
    for batch in loader:
        # each batch should be a dict or tuple depending on your BaseDataset
        # here we check the common structure
        assert "patient" in batch
        assert "time" in batch
        assert "data" in batch
        assert "target" in batch

        # check events can be loaded
        edf_path = batch["patient"][0]
        start_dt = pd.Timestamp(batch["time"][0])
        df = ds.get_event_df(edf_path, start_dt)
        assert not df.empty

        if ds.has_extra_target():
            assert "target_extra" in batch

            extra_df = ds.get_extra_event_df(edf_path, start_dt)
            assert not extra_df.empty

        # break early – we don’t need the whole dataset
        break