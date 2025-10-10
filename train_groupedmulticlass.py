#!/bin/env python3

from functools import partial
import logging
import os
import torch
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader

from torchinfo import summary
from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets import ChannelConfig
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.utils import get_edf_files_in_repo, kfold_split
from sleepwalker.datasets import Ruhrlandklinik

from sleepwalker.models import SleepTransformer
from sleepwalker.trainer.GroupedChanelMulticlassTrainer import GroupedChanelMulticlassTrainer
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.utils import append_to_jsonl
from sleepwalker.utils import logger, MlflowSink

""" Some info about logging

Sleepwalker uses a unified logging interface to centralize all logging and progress reporting.
Instead of sprinkling print(), logging, tqdm, or mlflow.log_* calls across the codebase, you interact with a single UnifiedLogger API.
This logger delegates events to one or more sinks, which are responsible for handling the actual output.
Each sink implements a common interface (event, metric, artifact, progress etc.), so you can mix and match outputs without changing your code.

Key Concepts
- Sinks 
    - StdLogSink: writes logs to console and file, using Python’s logging.
    - TqdmSink: provides progress bars using tqdm.
    - MlflowSink: integrates with MLflow for experiment tracking.
    - You can add your own sinks (e.g. GUI, database, HTTP endpoint).
- Contexts
    - Provide hierarchical prefixes for logs.
      Example: logger.context("XVAL 3/5") → all logs inside that block are tagged accordingly.
    - Contexts are also passed as tags to sinks that support them (e.g. MLflow).
- Progress
    - Instead of calling tqdm directly, use logger.progress_start(...), logger.progress_advance(...), and logger.progress_close().
    - This keeps progress reporting consistent across sinks (console, GUI, etc.).
- Metrics & Artifacts
    - logger.metric(name, value) → log scalar values (accuracy, loss, …).
    - logger.artifact(path, dest) → log files (models, configs, …).
    - logger.figure(name, fig) → log plots (matplotlib, …).

Default Setup - By default, get_logger() configures two sinks:
- StdLogSink → logs to console (with tqdm-aware output) and to a file (sleepwalker.log). Uses append mode by default. If you want a clean file, remove it before calling get_logger().
- TqdmSink → for progress bars. Shares the same formatting style as console logs for alignment.

Some examples for usage

Import: 
    from sleepwalker.utils import get_logger
    logger = get_logger()

Simple logging:
    logger.info("Starting experiment")
    logger.warning("EDF file missing metadata")
    logger.error("Training crashed due to NaN loss")
Expected Output
    2025-09-26 12:00:00,123 | INFO     | Starting experiment
    2025-09-26 12:00:01,456 | WARNING  | train.py:42 | EDF file missing metadata
    2025-09-26 12:00:02,789 | ERROR    | trainer.py:128 | Training crashed due to NaN loss

Context logging:
    logger.push_context("XVAL 3/5")
    logger.info("Loading dataset")

    logger.push_context("WARMUP")
    logger.info("Starting warmup")
    logger.pop_context()  # end WARMUP

    logger.pop_context()  # end XVAL 3/5
Expected Output:
    2025-09-26 12:01:00,000 | INFO     | XVAL 3/5 | Loading dataset
    2025-09-26 12:01:01,000 | INFO     | XVAL 3/5 | WARMUP | Starting warmup

Note: In MLflow, contexts are automatically turned into run tags. We currently do not support context managers, but you have to manually control contexts

Progress Bars:
    n_batches = len(data_loader)
    logger.progress_start(total=n_batches, desc="WARMUP")

    for batch in data_loader:
        loss = train_step(batch)
        logger.progress_status(f"Loss={loss:.4f}")
        logger.progress_advance(1)

    logger.progress_close()
Expected Output:
    2025-09-26 12:02:00,000 | INFO     | WARMUP |  37%|███████▉ | 91/243 [00:04<00:07, 19.0it/s]

Metrics:
    logger.metric("epoch/train/loss", 0.1234)
    logger.metric("epoch/train/accuracy", 0.89)
    logger.metric("batch/val/loss", 0.42)
Expected Output:
    Console/file: metrics are printed like regular logs.
    MLflow: metrics are stored in the current run, organized by name.
    If you want to add metrics to the progress bar, use progress_status(...)

Artifacts:
    path = save_checkpoint(model, optimizer, scheduler)
    logger.artifact(f"{path}/model.pt", dest="checkpoints/epoch-10")
    logger.artifact(f"{path}/optimizer.pt", dest="checkpoints/epoch-10")

Figures:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(history["loss"])
    logger.figure("loss_curve", fig)

MLFLow integration:
    logger.start_run(run_name="fold-0", params={"lr": 1e-3}, tags={"phase": "warmup"})
    logger.metric("epoch/train/loss", 0.123)
    logger.end_run()
"""

# Parameters for this run
edf_folder = "/raid/data/ruhrlandklinik/raw/train-test-2023"
n_splits = 5
#batch_size = 512
batch_size = 128
epochs = 100
total_input = "630s"
target_resolution = "30s"
n_samples = None #10_000
tracking_uri = "file:./mlruns"
experiment_name = "ruhrland_sleeptransformer"
groups = [["C4-M1"]]
# groups = [["C4-M1", "E2-M1", "F4-M1", "O2-M1"]]

# Function that is called by the Basedataset to filter out patients _before_ loading them 
# If true is returned => try to load patient
# If false is returned => try not to load patient
def filter_patient(edf_path):
    meta = read_edf_meta(edf_path)
    it = iter(meta["signals"])
    # Check if a patient has at-least 2 channels
    return all(any(it) for _ in range(2)) 

def build_loader(patients):
    dataset = Ruhrlandklinik(
        channels = [ChannelConfig(name=c, normalizer=None) for g in groups for c in g],
        patients = patients,
        num_workers = 8,
        sample_frequency = 100,
        event_mapping = {
            "wach": "wake",
            "n1": "n1",
            "n2": "n2",
            "n3": "n3",
            "rem": "rem"
        },
        filter_patient = filter_patient,
        get_item = partial(GroupedChanelMulticlassTrainer.get_item, groups=groups), # Implements 50% rule for multi-class classification
        online_filtering = True, # Let the dataset reject data points that are not usable for training by calling get_item
        total_input = total_input, 
        target_resolution = target_resolution
    )

    sampler = RandomSampler(dataset, num_samples = n_samples) if n_samples is not None else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=8, pin_memory=True, collate_fn=batch_collate, drop_last=False, persistent_workers=True)

    return loader, dataset

all_patients = get_edf_files_in_repo(edf_folder, recursive=False)

# Logging is now vastly simplified: 
#   We have a global singleton logger (from sleepwalker.utils import logger) that can be used for {metric,artifacts,text,...} logging
#   We can add sinks for logging to different backends (file, mlflow, etc). Per default we log to file and TQDM/Console
if os.path.exists("sleepwalker.log"):
    # Reset log file, per default we always append to the current file
    os.remove("sleepwalker.log")

# Add an mlflow sink with the appropriate experiment name and backend URI
# logger.add_sink(MlflowSink(tracking_uri="sqlite:///mlflow.sqlite", experiment="MyExperiment")) # can also be file:... as backend

for i, (train_patients, test_patients) in enumerate(kfold_split(all_patients, n_splits=n_splits)):
    logger.start_run(run_name=f"XVAL {i}") # Set the run_name for this experiment

    train_loader, dataset = build_loader(train_patients)

    model = SleepTransformer(classes=dataset.get_classes(), n_channels=len(groups))
    model_stats = summary(model, input_size=(1, dataset.get_timeseries_len(), len(groups)), depth=5, row_settings=["hide_recursive_layers"])

    trainer = GroupedChanelMulticlassTrainer(
        groups=groups,
        epochs=epochs, 
        optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=50),
        classes = dataset.get_classes(), 
        save_every = 10,
        loss_function=torch.nn.functional.cross_entropy, 
        n_apply_repeats_train=1
    )
    losses, cms = trainer.fit(model, train_loader)

    test_loader, _ = build_loader(test_patients)
    for r in [1,2,3,4,5,10]:
        logger.context(f"r={r}")
        test_loss, test_cm = trainer.test(model, test_loader, n_apply_repeats=r)

        record = {
            "xval":i,
            "repeat":r,
            "test_loss":test_loss,
            "test_cm":test_cm,
            "train_loss":losses,
            "train_cm":cms,
            "classes":dataset.get_classes(),
        }
        append_to_jsonl(experiment_name, record)
        logger.uncontext()
        
    logger.end_run()

