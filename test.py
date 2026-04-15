import os
import numpy as np
import pandas as pd
import torch
import zarr
import copy
from collections import defaultdict
import multiprocessing
from pathlib import Path
from numcodecs import Blosc

from tqdm import tqdm
from sleepwalker.core.signal import edf_to_df
from sleepwalker.datasets.SleepEDFx import SleepEDFx
from sleepwalker.datasets.ZarrDataset import dataset_to_zarr
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.utils import logger
from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.utils import dataset_to_numpy, get_edf_files_in_repo

# def save_lists_to_hdf(filename: str, dfs, ys):
#     with pd.HDFStore(filename, mode="w") as store:
#         for i, df in enumerate(dfs):
#             store.put(f"df_{i}", df, format="table")
#         for i, y in enumerate(ys):
#             store.put(f"y_{i}", y, format="table")

# edf_files = get_edf_files_in_repo("/raid/sleepwalker/sleep-edfx", recursive=True)
# edf_files = [e for e in edf_files if "Hypnogram" not in e]

# dataset = SleepEDFx(
#     channels = [
#         ChannelConfig(name=ci, normalizer=EEGFilterNormalizer(fs=100)) for ci in ["EEG Fpz-Cz", "EEG Fpz-Cz"]
#     ],
#     patients = edf_files,
#     num_workers = 8,
#     sample_frequency = 100,
#     event_mapping = {   
#         "sleep stage w": "wake",
#         "sleep stage 1": "n1",
#         "sleep stage 2": "n2",
#         "sleep stage 3": "n3",
#         "sleep stage 4": "n3",
#         "sleep stage r": "rem"
#     },
#     get_item = None,
#     total_input = "120s", 
#     target_resolution = "30s",
#     transform = None,
#     online_max_tries = 0
# )

# dfs = []
# ys = []
# for e in tqdm(edf_files, total=len(edf_files), desc="Parsing edf files"):
#     try:
#         df = edf_to_df(e, channels=["EEG Fpz-Cz", "EEG Fpz-Cz"], frequency=100, start=None, end=None)
#         y = dataset.get_event_df(e, None)
#         dfs.append(df)
#         ys.append(y)
#     except:
#         pass
    
# save_lists_to_hdf("test.hdf", dfs, ys)


edf_files = get_edf_files_in_repo("/raid/sleepwalker/shhs", ending="edf", recursive=True)
print(len(edf_files))