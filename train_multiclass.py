#!/bin/env python3

import argparse
import os
from functools import partial

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torchinfo import summary

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.utils import estimate_class_cnts, get_edf_files_in_repo, random_split
from sleepwalker.trainer.losses import class_weights_for_loss
from sleepwalker.trainer.tasks.sleep_staging import DATASET_CFG, MODEL_CFG, TARGET_CLASSES, get_dataset, get_model_and_trainer
from sleepwalker.trainer.utils import append_to_jsonl, get_target_as_multiclass, trim_wake
from sleepwalker.utils import logger, suppress_stdout_logging

from lamarr_energy_tracker import EnergyTracker

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

mp.set_sharing_strategy("file_system")

SCORED_SLEEP_LABELS = ["n1", "n2", "n3", "rem"]
BATCH_SIZE = 128
N_SAMPLES = 250_000
NUM_WORKERS_DATASET = 16
NUM_WORKERS_DATALOADER = 16
TEST_FRAC = 0.3
VAL_FRAC = 0.1
EXPERIMENT_NAME = "multiclass"
DATASET_ROOT = "/raid/sleepwalker"
SLEEP_PERCENTAGE = 0.5
REJECT_OUTSIDE_SLEEP = False


def filter_multiclass_window(
    target,
    target_extra=None,
    reject_outside_sleep: bool = False,
    sleep_percentage: float = 0.5,
    **_kwargs,
):
    if target is None:
        return False

    if reject_outside_sleep:
        sleep_cols = [c for c in SCORED_SLEEP_LABELS if c in target.columns]
        if len(sleep_cols) == 0:
            raise ValueError(
                f"reject_outside_sleep requires sleep labels {SCORED_SLEEP_LABELS}, "
                f"but target columns are {list(target.columns)}."
            )
        if target[sleep_cols].any(axis=1).mean() < sleep_percentage:
            return False

    return True


def build_multiclass_target(
    target,
    target_extra=None,
    percentage: float = 0.5,
    class_cnts=None,
    target_classes=None,
    **_kwargs,
):
    if target is None:
        return None

    if target_classes is not None:
        target = target.reindex(columns=target_classes, fill_value=0)
        if target_extra is not None:
            target_extra = target_extra.reindex(columns=target_classes, fill_value=0)

    return {
        "target": get_target_as_multiclass(
            target=target,
            target_extra=target_extra,
            percentage=percentage,
            class_cnts=class_cnts,
        )
    }


def build_multiclass_sample(data, target, **item):
    item["data"] = torch.from_numpy(data.values).float()
    item["target"] = target
    return item


def get_dataset_splits(dataset_name: str, model_name: str, grouped: bool, target_classes: list[str]):
    dataset_path = os.path.join(DATASET_ROOT, DATASET_CFG[dataset_name]["edf_path"])
    patients = get_edf_files_in_repo(dataset_path, recursive=True)

    train_patients, test_patients = random_split(patients, test_frac=TEST_FRAC)
    train_patients, val_patients = random_split(train_patients, test_frac=VAL_FRAC)

    filter_window = partial(
        filter_multiclass_window,
        reject_outside_sleep=REJECT_OUTSIDE_SLEEP,
        sleep_percentage=SLEEP_PERCENTAGE,
    )
    build_target = partial(build_multiclass_target, target_classes=target_classes)

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_ds = get_dataset(
            name=dataset_name,
            model_name=model_name,
            patients=train_patients,
            path_root=DATASET_ROOT,
            grouped=grouped,
            num_workers=NUM_WORKERS_DATASET,
            filter_target=trim_wake,
            filter_window=filter_window,
            build_target=build_target,
            build_sample=build_multiclass_sample,
        )
        logger.uncontext()
        logger.context("VAL")
        val_ds = get_dataset(
            name=dataset_name,
            model_name=model_name,
            patients=val_patients,
            path_root=DATASET_ROOT,
            grouped=grouped,
            num_workers=NUM_WORKERS_DATASET,
            filter_target=trim_wake,
            filter_window=filter_window,
            build_target=build_target,
            build_sample=build_multiclass_sample,
        )
        logger.uncontext()
        logger.context("TEST")
        test_ds = get_dataset(
            name=dataset_name,
            model_name=model_name,
            patients=test_patients,
            path_root=DATASET_ROOT,
            grouped=grouped,
            num_workers=NUM_WORKERS_DATASET,
            filter_target=trim_wake,
            filter_window=filter_window,
            build_target=build_target,
            build_sample=build_multiclass_sample,
        )
        logger.uncontext()

    logger.info(
        f"Loaded {train_ds.get_n_patients()} / {len(train_patients)} patients for training and "
        f"{test_ds.get_n_patients()} / {len(test_patients)} patients for test and "
        f"{val_ds.get_n_patients()} / {len(val_patients)} patients for validation"
    )
    return train_ds, val_ds, test_ds


def run(model_name, dataset_name, dry_run, grouped=False):
    if dataset_name not in DATASET_CFG:
        logger.warning(f"Unknown sleep staging dataset '{dataset_name}'.")
        return
    if model_name not in MODEL_CFG:
        logger.warning(f"Unknown sleep staging model '{model_name}'.")
        return

    target_classes = TARGET_CLASSES

    try:
        train_ds, val_ds, test_ds = get_dataset_splits(
            dataset_name=dataset_name,
            model_name=model_name,
            grouped=grouped,
            target_classes=target_classes,
        )
    except ValueError as exc:
        logger.warning(str(exc))
        return

    model, trainer, run_cfg = get_model_and_trainer(
        name=model_name,
        dataset=train_ds,
        dry_run=dry_run,
        grouped=grouped,
    )

    run_cfg["batch_size"] = BATCH_SIZE
    run_cfg["n_samples"] = 1_000 if dry_run else N_SAMPLES
    experiment_name = f"{EXPERIMENT_NAME}-grouped" if grouped else EXPERIMENT_NAME
    if dry_run:
        logger.info("Performing dry run to test pipeline!")
        experiment_name += "-dev"
    else:
        tracker = EnergyTracker(project_name=experiment_name)
        tracker.start()

    logger.start_run(run_name=experiment_name, tags={"model": model_name})

    if os.path.exists("sleepwalker.log"):
        os.remove("sleepwalker.log")

    logger.info(f"Loaded {train_ds.get_n_patients()} for training")
    logger.info(f"Loaded {val_ds.get_n_patients()} for validation")
    logger.info(f"Loaded {test_ds.get_n_patients()} for testing")
    logger.info(f"Input data is {run_cfg['ts_len']} x {run_cfg['n_channels']}")
    summary(model, input_size=(1, run_cfg["ts_len"], run_cfg["n_channels"]), depth=5, row_settings=["hide_recursive_layers"])

    class_cnts = None
    if run_cfg["balance_batches"]:
        class_cnts = estimate_class_cnts(
            train_ds,
            run_cfg["n_samples"],
            NUM_WORKERS_DATALOADER,
            run_cfg["batch_size"],
        )
        class_cnts_list = [class_cnts.get(c, 1) for c in train_ds.get_classes()]
        train_ds.build_target_callback = partial(
            build_multiclass_target,
            target_classes=target_classes,
            class_cnts=class_cnts_list,
        )

    if run_cfg["loss_mode"] != "regular":
        if class_cnts is None:
            class_cnts = estimate_class_cnts(
                train_ds,
                run_cfg["n_samples"],
                NUM_WORKERS_DATALOADER,
                run_cfg["batch_size"],
            )
        class_weights = class_weights_for_loss(run_cfg["class_weights"], class_cnts, run_cfg["loss_mode"])
    else:
        class_weights = run_cfg["class_weights"]

    if len(class_weights) > 0:
        weights_torch = torch.tensor([class_weights.get(c, 1) for c in train_ds.get_classes()], device="cuda")
        trainer.loss_function = partial(run_cfg["loss_function"], weight=weights_torch)

    train_sampler = RandomSampler(train_ds, num_samples=run_cfg["n_samples"]) if run_cfg["n_samples"] is not None else None
    val_sampler = RandomSampler(val_ds, num_samples=run_cfg["n_samples"]) if run_cfg["n_samples"] is not None else None

    train_loader = DataLoader(
        train_ds,
        batch_size=run_cfg["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=NUM_WORKERS_DATALOADER,
        collate_fn=partial(batch_collate, ignore_list=["time", "patient"]),
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=run_cfg["batch_size"],
        shuffle=val_sampler is None,
        sampler=val_sampler,
        num_workers=NUM_WORKERS_DATALOADER,
        collate_fn=partial(batch_collate, ignore_list=["time", "patient"]),
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
    )

    train_dict = trainer.fit(model, train_loader, val_loader)
    losses, cms = train_dict["losses"], train_dict["cms"]
    if "checkpoint" in train_dict:
        with torch.inference_mode():
            state_dict = torch.load(os.path.join(train_dict["checkpoint"], "model.pt"), map_location="cpu")
            model.load_state_dict(state_dict)

    GROUPED_TEST_REPEATS = [1, 2, 3, 4, 5, 10] if else [1]
    for repeat in GROUPED_TEST_REPEATS:
        if grouped:
            trainer.n_repeat_test = repeat
            logger.context(f"r={repeat}")

        test_loader = DataLoader(
            test_ds,
            batch_size=run_cfg["batch_size"],
            shuffle=False,
            sampler=None,
            num_workers=NUM_WORKERS_DATALOADER,
            collate_fn=partial(batch_collate, ignore_list=["time", "patient"]),
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True,
        )
        test_loss, test_cm = trainer.test(model, test_loader)
        record = {
            "model": model_name,
            "test_loss": test_loss,
            "test_cm": test_cm,
            "train_loss": losses,
            "train_cm": cms,
            "dataset": dataset_name,
            "classes": train_ds.get_classes(),
            "artifact_root": train_dict["checkpoint"] if "checkpoint" in train_dict else "",
        }
        if grouped:
            record["repeat"] = repeat
        if "best_model" in train_dict:
            record["best_model"] = train_dict["best_model"]
        append_to_jsonl(experiment_name, record)
        if grouped:
            logger.uncontext()

    if not dry_run:
        tracker.stop()
    logger.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a sleep-staging model on one dataset.")
    parser.add_argument("--model", required=False, default="sleeptransformer", type=str)
    parser.add_argument("--data", required=False, default="sleepedfx", type=str)
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    run(args.model, args.data, args.dry, grouped=args.grouped)
