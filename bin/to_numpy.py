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
from sleepwalker.datasets.SleepEDFx import SleepEDFx
from sleepwalker.datasets.ZarrDataset import dataset_to_zarr
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.utils import logger
from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.utils import dataset_to_numpy, get_edf_files_in_repo

for fs in [100, 200]:
    for c in [["EEG Fpz-Cz"], ["EEG Pz-Oz"], ["EEG Fpz-Cz", "EEG Fpz-Cz"]]:
        for ti in ["30s", "90s", "120s"]:
            edf_files = get_edf_files_in_repo("/raid/sleepwalker/sleep-edfx", recursive=True)
            edf_files = [e for e in edf_files if "Hypnogram" not in e]

            dataset = SleepEDFx(
                channels = [
                    ChannelConfig(name=ci, normalizer=EEGFilterNormalizer(fs=fs)) for ci in c
                ],
                patients = edf_files,
                num_workers = 8,
                sample_frequency = fs,
                event_mapping = {   
                    "sleep stage w": "wake",
                    "sleep stage 1": "n1",
                    "sleep stage 2": "n2",
                    "sleep stage 3": "n3",
                    "sleep stage 4": "n3",
                    "sleep stage r": "rem"
                },
                get_item = MulticlassTrainer.get_item,
                total_input = ti, 
                target_resolution = "30s",
                transform = None,
                online_max_tries = 0
            )

            logger.context(f"fs={fs}Hz, c={c}, ti={ti}")
            X, Y, Y_extra, timestamps, patients = dataset_to_numpy(dataset, n_samples = None, num_workers = 8, batch_size = 128)
            logger.uncontext()
            
            if len(c) == 2:
                c_name = "fpz_cz+pz_oz"
            else:
                c_name = "fpz_cz" if "Fpz-Cz" in c[0] else "pz_oz" 
            logger.info(f"Data shape is: {X.shape}")
            logger.info(f"Target shape is: {Y.shape}")

            filename = f"sleepedfx_{c_name}_{fs}Hz_{ti}"
            np.save(filename+"_X.npy", X, allow_pickle=False)
            np.save(filename+"_Y.npy", Y, allow_pickle=False)



 