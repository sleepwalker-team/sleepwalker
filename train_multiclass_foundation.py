#!/bin/env python3

import argparse
import os

import mlflow

from sleepwalker.trainer.NegativeGroupedChanelMulticlassTrainer import GradReverseTrainer

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from functools import partial
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.multiprocessing as mp

from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.utils import estimate_class_cnts, get_edf_files_in_repo, random_split
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.losses import class_weights_for_loss
from sleepwalker.trainer.tasks.sleep_staging import (
    build_dataset_instance,
    build_model_instance,
    resolve_task_config,
)
from sleepwalker.trainer.utils import append_to_jsonl
from sleepwalker.utils import logger, MlflowSink, suppress_stdout_logging

from lamarr_energy_tracker import EnergyTracker

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

mp.set_sharing_strategy('file_system')

def build_dataset(dataset_config: dict, total_input: str, num_workers_dataset: int, get_target_fn, test_frac: Optional[float] = 0.1, transform = None, dry_run:bool = False):
    edf_path = dataset_config["edf_path"]
    edf_files = get_edf_files_in_repo(edf_path, recursive=True)
    if dry_run:
        edf_files = edf_files[:2]
        test_frac = 0.5

    if not test_frac or test_frac <= 0:
        with suppress_stdout_logging(logger):
            train_ds = build_dataset_instance(
                config=dataset_config,
                patients=edf_files,
                num_workers=num_workers_dataset,
                get_target=get_target_fn,
                total_input=total_input,
                transform=transform,
            )
        test_ds = None
        logger.info(f"Loaded {train_ds.get_n_patients()} / {len(edf_files)} patients.")
    else:
        train_patients, test_patients = random_split(edf_files, test_frac=test_frac)
        with suppress_stdout_logging(logger):
            logger.context("TRAIN")
            train_ds = build_dataset_instance(
                config=dataset_config,
                patients=train_patients,
                num_workers=num_workers_dataset,
                get_target=get_target_fn,
                total_input=total_input,
                transform=transform,
            )
            logger.uncontext()
            logger.context("TEST")
            test_ds = build_dataset_instance(
                    config=dataset_config,
                    patients=test_patients,
                    num_workers=num_workers_dataset,
                    get_target=get_target_fn,
                    total_input=total_input,
                    transform=transform,
                )
            logger.uncontext()
    
        logger.info(f"Loaded {train_ds.get_n_patients()} / {len(train_patients)} patients for training and {test_ds.get_n_patients()} / {len(test_patients)} patients for validation")
    return train_ds, test_ds

def run(model_name, train_datasets, test_datasets, gradient_reversal, dry_run):
    dataset_root = "/raid/sleepwalker"

    try:
        model_resolved_cfg = resolve_task_config(
            task="sleep_staging",
            dataset={"name": train_datasets[0]},
            model={"name": model_name},
            path_root=dataset_root,
        )
    except ValueError as exc:
        logger.warning(str(exc))
        return

    trainer_config = dict(model_resolved_cfg["trainer"])
    trainer_config["experiment_name"] = "multiclass-foundation"
    batch_size = trainer_config["batch_size"]
    epochs = trainer_config["epochs"] if not dry_run else 1
    n_samples = trainer_config["n_samples"] if not dry_run else 1_000
    experiment_name = trainer_config["experiment_name"]
    num_workers_dataset = trainer_config["num_workers_dataset"]
    num_workers_dataloader = trainer_config["num_workers_dataloader"]

    model_config = model_resolved_cfg["model"]
    total_input = model_config["total_input"]
    lr_scheduler = model_config["lr_scheduler"]
    optimizer = model_config["optimizer"]
    loss_function = model_config["loss_function"]
    loss_mode = model_config.get("loss_mode", "regular")
    class_weights = model_config.get("class_weights", {})
    transform = model_config.get("transform", None)
    balance_batches = model_config.get("balance_batches", False)

    if dry_run:
        logger.info("Performing dry run to test pipeline! Each dataset loads at most 2 patients")
        experiment_name += "-dev"
    else:
        tracker = EnergyTracker(project_name=experiment_name)
        tracker.start()
    
    # with open(os.path.expanduser("~/mlflow/auth_config.ini"),"r") as f:
    #     TRACKING_URI=f.read().strip()
    # ARTIFACT_URI=f"/raid/mlruns"

    # mlflow_sink = MlflowSink(tracking_uri=TRACKING_URI, experiment=experiment_name, artifact_uri=ARTIFACT_URI)
    # logger.add_sink(mlflow_sink)
    logger.start_run(run_name=experiment_name,tags={"model":model_name})

    train_ds = []
    val_ds = []

    for ds in train_datasets:
        try:
            resolved_cfg = resolve_task_config(
                task="sleep_staging",
                dataset={"name": ds},
                model={"name": model_name},
                trainer={"experiment_name": experiment_name},
                path_root=dataset_root,
            )
        except ValueError as exc:
            logger.warning(str(exc))
            continue

        dataset_config = resolved_cfg["dataset"]
        logger.context(ds)
        train, val = build_dataset(
            dataset_config=dataset_config,
            total_input=total_input,
            get_target_fn=MulticlassTrainer.get_target,
            num_workers_dataset=num_workers_dataset,
            test_frac=trainer_config["val_frac"],
            transform=transform,
            dry_run=dry_run,
        )
        logger.uncontext()
        train_ds.append(train)
        val_ds.append(val)

    test_ds = []
    for ds in test_datasets:
        try:
            resolved_cfg = resolve_task_config(
                task="sleep_staging",
                dataset={"name": ds},
                model={"name": model_name},
                trainer={"experiment_name": experiment_name},
                path_root=dataset_root,
            )
        except ValueError as exc:
            logger.warning(str(exc))
            continue
        dataset_config = resolved_cfg["dataset"]
        logger.context(ds)
        test, _ = build_dataset(
            dataset_config=dataset_config,
            # TinySleepNet uses augmentation that changes the model input size. For testing we have to be consistent
            total_input=model_config["total_input_model"] if "total_input_model" in model_config else total_input,
            get_target_fn=MulticlassTrainer.get_target,
            num_workers_dataset=num_workers_dataset,
            test_frac=0,
            dry_run=dry_run,
        )
        logger.uncontext()
        test_ds.append(test)

    if os.path.exists("sleepwalker.log"):
        os.remove("sleepwalker.log")

    train_multi_ds = MultiDataset(train_ds)
    logger.info(f"Loaded {train_multi_ds.get_n_patients()} for training")

    val_multi_ds = MultiDataset(val_ds)
    logger.info(f"Loaded {val_multi_ds.get_n_patients()} for validation")

    test_multi_ds = MultiDataset(test_ds)
    logger.info(f"Loaded {test_multi_ds.get_n_patients()} for testing")

    if "total_input_model" in model_config:
        # TinySleepNet uses augmentation that changes the model input size. Hence its actual input size is different from total_input 
        freq = pd.to_timedelta(1.0 / train_multi_ds.sample_frequency, unit="s")
        total_input = pd.to_timedelta(model_config["total_input_model"])
        ts_len = int(total_input.total_seconds() / freq.total_seconds())
    else:
        ts_len = train_multi_ds.get_timeseries_len()

    n_channels = len(train_multi_ds.datasets[0].channels)
    model = build_model_instance(
        config=model_config,
        classes=train_multi_ds.get_classes(),
        ts_len=ts_len,
        n_channels=n_channels,
        sampling_frequency=train_multi_ds.sample_frequency,
    )
    logger.info(f"Input data is {ts_len} x {n_channels}")
    model_stats = summary(model, input_size=(1, ts_len, n_channels), depth=5, row_settings=["hide_recursive_layers"])

    if balance_batches:
        class_cnts = estimate_class_cnts(train_multi_ds, n_samples, num_workers_dataloader, batch_size)
        class_cnts_list = [class_cnts.get(c, 1) for c in train_multi_ds.get_classes()]
        for i in range(len(train_datasets)):
            train_multi_ds.datasets[i].get_target_callback = partial(MulticlassTrainer.get_target, class_cnts=class_cnts_list)
    else:
        class_cnts = None

    if loss_mode != "regular":
        if class_cnts is None: 
            class_cnts = estimate_class_cnts(train_multi_ds, n_samples, num_workers_dataloader, batch_size) 
        class_weights = class_weights_for_loss(class_weights, class_cnts, loss_mode)
    
    if len(class_weights) > 0:
        weights_torch = torch.tensor([class_weights.get(c, 1) for c in train_multi_ds.get_classes()], device="cuda")
    else:
        weights_torch = None

    if gradient_reversal:
        trainer = GradReverseTrainer(
            feature_dim = 512,
            n_domains = train_multi_ds.get_n_datasets(),
            epochs=epochs, 
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            classes = train_multi_ds.get_classes(), 
            save_every = 1,
            loss_function=partial(loss_function, weight=weights_torch), 
            early_stopping = 10
        )
    else:
        trainer = MulticlassTrainer(
            epochs=epochs, 
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            classes = train_multi_ds.get_classes(), 
            save_every = 1,
            loss_function=partial(loss_function, weight=weights_torch), 
            early_stopping = 10
        )
    train_sampler = RandomSampler(train_multi_ds, num_samples = n_samples) if n_samples is not None else None
    train_loader = DataLoader(train_multi_ds, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=num_workers_dataloader, collate_fn=partial(batch_collate, ignore_list=["time", "patient", "dataset"]), drop_last=False, persistent_workers=True, prefetch_factor=2, pin_memory=True) ### prefetch_factor=12

    val_sampler = RandomSampler(val_multi_ds, num_samples = n_samples) if n_samples is not None else None
    val_loader = DataLoader(val_multi_ds, batch_size=batch_size, shuffle=val_sampler is None, sampler=val_sampler, num_workers=num_workers_dataloader, collate_fn=partial(batch_collate, ignore_list=["time", "patient", "dataset"]), drop_last=False, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    train_dict = trainer.fit(model, train_loader, val_loader)
    losses, cms = train_dict["losses"], train_dict["cms"]
    if "checkpoint" in train_dict:
        with torch.inference_mode():
            state_dict = torch.load(os.path.join(train_dict["checkpoint"], "model.pt"), map_location="cpu")
            model.load_state_dict(state_dict)
    
    for t_ds, name in zip(test_ds, test_datasets):
        test_loader = DataLoader(t_ds, batch_size=batch_size, shuffle=False, sampler=None, num_workers=num_workers_dataloader, collate_fn=partial(batch_collate, ignore_list=["time", "patient", "dataset"]), drop_last=False, persistent_workers=True, prefetch_factor=2, pin_memory=True)

        logger.context(f"{name}")
        test_loss, test_cm = trainer.test(model, test_loader)

        record = {
            "model": f"{model_name}{'_GradReversal' if gradient_reversal else ''}",
            "test_loss":test_loss,
            "test_cm":test_cm,
            "train_loss":losses,
            "train_cm":cms,
            "dataset":name,
            "classes":train_multi_ds.get_classes(),
            "artifact_root": train_dict["checkpoint"] if "checkpoint" in train_dict else "",
        }
        if "best_model" in train_dict:
            record["best_model"] = train_dict["best_model"]

        append_to_jsonl(f"{experiment_name}_{model_name}", record)
        logger.uncontext()

    if not dry_run: tracker.stop()
    logger.end_run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a sleep-staging model on multiple datasets with a single EEG channels.')
    parser.add_argument("--model", help='The model to be trained. Appropriate training parameters are choosen automatically', required=False, default="sleeptransformer", type=str)
    parser.add_argument("--train", help='Datasets to train the model on', required=False, default=["abc", "cap", "isruc", "mnc", "nchsdb", "svuhucd", "shhs"], type=str, nargs="+")
    parser.add_argument("--test", help='Datasets to test the model on', required=False, default=["apples", "mros", "ruhrland", "sleepedfx"], type=str, nargs="+")
    parser.add_argument("--gradrev", help='Dataset to train the model on', action="store_true")
    parser.add_argument("--dry", help='If set, performs a quick test run without actually training the model', action="store_true")
    args = parser.parse_args()
    run(args.model, args.train, args.test, args.gradrev, args.dry)
