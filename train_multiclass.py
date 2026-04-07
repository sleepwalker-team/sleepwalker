#!/bin/env python3

import argparse
import os
from functools import partial

import pandas as pd
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

from sleepwalker.datasets.Basedataset import BaseDataset, batch_collate
from sleepwalker.datasets.MultiDataset import MultiDataset
from sleepwalker.datasets.utils import estimate_class_cnts, get_edf_files_in_repo, random_split
from sleepwalker.trainer.losses import class_weights_for_loss
from sleepwalker.trainer.tasks.sleep_staging import DATASET_CFG, MODEL_CFG, TARGET_CLASSES, get_dataset, get_model_and_trainer
from sleepwalker.trainer.utils import append_to_jsonl, build_multiclass_target, trim_wake
from sleepwalker.utils import logger, suppress_stdout_logging

from lamarr_energy_tracker import EnergyTracker

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
mp.set_sharing_strategy("file_system")

SCORED_SLEEP_LABELS = ["n1", "n2", "n3", "rem"]
BATCH_SIZE = 128
EPOCHS = 100
N_SAMPLES = 250_000
NUM_WORKERS_DATASET = 16
NUM_WORKERS_DATALOADER = 16
TEST_FRAC = 0.3
VAL_FRAC = 0.1
EXPERIMENT_NAME = "multiclass"
DATASET_ROOT = "/raid/sleepwalker"
SLEEP_PERCENTAGE = 0.5
REJECT_OUTSIDE_SLEEP = False #only used when we not do sleep staging. to be implemented
GROUPED_TEST_REPEATS = [1, 2, 3, 4, 5, 10]
SLEEP_TIME_FILTER_QUANTILE = 0.05


def prepare_multiclass_target(
    target,
    target_extra=None,
    patient=None,
    time=None,
    percentage: float = 0.5,
    class_cnts=None,
    target_classes=None,
):
    if target is None:
        return None

    if REJECT_OUTSIDE_SLEEP:
        sleep_cols = [c for c in SCORED_SLEEP_LABELS if c in target.columns]
        if len(sleep_cols) == 0:
            raise ValueError(
                f"reject_outside_sleep requires sleep labels {SCORED_SLEEP_LABELS}, "
                f"but target columns are {list(target.columns)}."
            )
        if target[sleep_cols].any(axis=1).mean() < SLEEP_PERCENTAGE:
            return None

    if target_classes is not None:
        target = target.reindex(columns=target_classes, fill_value=0)
        if target_extra is not None:
            target_extra = target_extra.reindex(columns=target_classes, fill_value=0)

    return build_multiclass_target(
        target=target,
        target_extra=target_extra,
        percentage=percentage,
        class_cnts=class_cnts,
    )

def prepare_sleep_staging_patient(data_df, label_df, label_extra_df, patient=None):
    trimmed = trim_wake(data_df, label_df, label_extra_df)
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df

# def prepare_multiclass_sample(data, target, **item):
#     item["data"] = torch.from_numpy(data.values).float()
#     item["target"] = target
#     if item.get("target_extra") is None:
#         item.pop("target_extra", None)
#     return item

def summarize_patient_sleep_time(patient: str, label_df: pd.DataFrame | None, **_kwargs) -> dict | None:
    if label_df is None or len(label_df) == 0:
        return None
    trimmed_events = label_df.copy()
    trimmed_events["duration_s"] = (trimmed_events["Endtime"] - trimmed_events["Starttime"]).dt.total_seconds()
    sleep_seconds = trimmed_events.loc[trimmed_events["Label"].isin(SCORED_SLEEP_LABELS), "duration_s"].sum()
    return {"patient": patient, "sleep_seconds": float(sleep_seconds)}


def filter_patients_by_sleep_time(dataset_name: str, model_name: str, grouped: bool, patients: list[str]) -> list[str]:
    if len(patients) < 3:
        return patients

    dataset = get_dataset(
        name=dataset_name,
        model_name=model_name,
        patients=[],
        path_root=DATASET_ROOT,
        grouped=grouped,
        prepare_patient=prepare_sleep_staging_patient,
    )
    stats_df = dataset.get_patient_stats(patients, summarize_patient_sleep_time, num_workers=NUM_WORKERS_DATASET)
    if len(stats_df) < 3 or "sleep_seconds" not in stats_df.columns:
        return patients

    lower = stats_df["sleep_seconds"].quantile(SLEEP_TIME_FILTER_QUANTILE)
    upper = stats_df["sleep_seconds"].quantile(1 - SLEEP_TIME_FILTER_QUANTILE)
    filtered = stats_df[ (stats_df["sleep_seconds"] >= lower) & (stats_df["sleep_seconds"] <= upper)]["patient"].tolist()
    logger.info(
        f"Filtered sleep-time outliers for {dataset_name}: kept {len(filtered)}/{len(patients)} patients "
        f"after dropping the bottom/top {SLEEP_TIME_FILTER_QUANTILE:.0%}."
    )
    return filtered


def get_collate_ignore_list(dataset) -> list[str]:
    return ["time", "patient", "dataset"] if isinstance(dataset, MultiDataset) else ["time", "patient"]


def build_loader(dataset, batch_size: int, n_samples: int | None, shuffle_default: bool):
    sampler = RandomSampler(dataset, num_samples=n_samples) if n_samples is not None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_default and sampler is None,
        sampler=sampler,
        num_workers=NUM_WORKERS_DATALOADER,
        collate_fn=partial(batch_collate, ignore_list=get_collate_ignore_list(dataset)),
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
    )

def build_dataset_from_patients(dataset_name: str, model_name: str, grouped: bool, patients: list[str]):
    prepare_target = partial(prepare_multiclass_target, target_classes=TARGET_CLASSES)

    dataset = get_dataset(
        name=dataset_name,
        model_name=model_name,
        patients=patients,
        path_root=DATASET_ROOT,
        grouped=grouped,
        prepare_patient=prepare_sleep_staging_patient,
        prepare_target=prepare_target,
        prepare_sample=None,
    )
    dataset.initialize(patients, NUM_WORKERS_DATASET)
    return dataset

def split_train_val_test(dataset_name: str, model_name: str, grouped: bool, dry_run: bool):
    dataset_path = os.path.join(DATASET_ROOT, DATASET_CFG[dataset_name]["edf_path"])
    patients = get_edf_files_in_repo(dataset_path, recursive=True)
    patients = filter_patients_by_sleep_time(dataset_name, model_name, grouped, patients)
    if dry_run:
        patients = patients[:2]

    train_patients, test_patients = random_split(patients, test_frac=TEST_FRAC)
    train_patients, val_patients = random_split(train_patients, test_frac=VAL_FRAC)

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_ds = build_dataset_from_patients(dataset_name, model_name, grouped, train_patients)
        logger.uncontext()
        logger.context("VAL")
        val_ds = build_dataset_from_patients(dataset_name, model_name, grouped, val_patients)
        logger.uncontext()
        logger.context("TEST")
        test_ds = build_dataset_from_patients(dataset_name, model_name, grouped, test_patients)
        logger.uncontext()
    return train_ds, val_ds, test_ds


def split_train_val(dataset_name: str, model_name: str, grouped: bool, dry_run: bool):
    dataset_path = os.path.join(DATASET_ROOT, DATASET_CFG[dataset_name]["edf_path"])
    patients = get_edf_files_in_repo(dataset_path, recursive=True)
    patients = filter_patients_by_sleep_time(dataset_name, model_name, grouped, patients)
    if dry_run:
        patients = patients[:2]

    train_patients, val_patients = random_split(patients, test_frac=VAL_FRAC)
    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_ds = build_dataset_from_patients(dataset_name, model_name, grouped, train_patients)
        logger.uncontext()
        logger.context("VAL")
        val_ds = build_dataset_from_patients(dataset_name, model_name, grouped, val_patients)
        logger.uncontext()
    return train_ds, val_ds


def load_test_dataset(dataset_name: str, model_name: str, grouped: bool, dry_run: bool):
    dataset_path = os.path.join(DATASET_ROOT, DATASET_CFG[dataset_name]["edf_path"])
    patients = get_edf_files_in_repo(dataset_path, recursive=True)
    patients = filter_patients_by_sleep_time(dataset_name, model_name, grouped, patients)
    if dry_run:
        patients = patients[:2]

    total_input = MODEL_CFG[model_name].get("total_input_model", MODEL_CFG[model_name]["total_input"])
    with suppress_stdout_logging(logger):
        logger.context("TEST")
        test_ds = get_dataset(
            name=dataset_name,
            model_name=model_name,
            patients=patients,
            path_root=DATASET_ROOT,
            grouped=grouped,
            prepare_patient=prepare_sleep_staging_patient,
            prepare_target=partial(prepare_multiclass_target, target_classes=TARGET_CLASSES),
            prepare_sample=prepare_multiclass_sample,
            total_input=total_input,
        )
        test_ds.initialize(patients, NUM_WORKERS_DATASET)
        logger.uncontext()
    return test_ds


def run(model_name, train_datasets: list[str], test_datasets: list[str] | None, dry_run: bool, grouped: bool = False, grad_reversal: bool = False):
    if any(name not in DATASET_CFG for name in train_datasets):
        unknown = sorted({name for name in train_datasets if name not in DATASET_CFG})
        logger.warning(f"Unknown sleep staging training dataset(s): {unknown}.")
        return
    if test_datasets and any(name not in DATASET_CFG for name in test_datasets):
        unknown = sorted({name for name in test_datasets if name not in DATASET_CFG})
        logger.warning(f"Unknown sleep staging test dataset(s): {unknown}.")
        return
    if model_name not in MODEL_CFG:
        logger.warning(f"Unknown sleep staging model '{model_name}'.")
        return
    if grad_reversal and len(train_datasets) < 3:
        logger.warning("Gradient reversal requires more than two training datasets.")
        return

    train_parts = []
    val_parts = []
    test_parts: list[tuple[str, BaseDataset]] = []

    if test_datasets:
        for dataset_name in train_datasets:
            logger.context(dataset_name)
            train_ds, val_ds = split_train_val(dataset_name, model_name, grouped, dry_run)
            logger.uncontext()
            train_parts.append(train_ds)
            val_parts.append(val_ds)

        for dataset_name in test_datasets:
            logger.context(dataset_name)
            test_ds = load_test_dataset(dataset_name, model_name, grouped, dry_run)
            logger.uncontext()
            test_parts.append((dataset_name, test_ds))
    else:
        for dataset_name in train_datasets:
            logger.context(dataset_name)
            train_ds, val_ds, test_ds = split_train_val_test(dataset_name, model_name, grouped, dry_run)
            logger.uncontext()
            train_parts.append(train_ds)
            val_parts.append(val_ds)
            test_parts.append((dataset_name, test_ds))

    train_dataset = MultiDataset(train_parts) if len(train_parts) > 1 else train_parts[0]
    val_dataset = MultiDataset(val_parts) if len(val_parts) > 1 else val_parts[0]

    model, trainer, run_cfg = get_model_and_trainer(
        name=model_name,
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        n_samples=N_SAMPLES,
        dry_run=dry_run,
        grad_reversal=grad_reversal,
    )

    run_cfg["batch_size"] = BATCH_SIZE
    run_cfg["n_samples"] = 1_000 if dry_run else N_SAMPLES
    experiment_name = f"{EXPERIMENT_NAME}-grouped" if grouped else EXPERIMENT_NAME
    if grad_reversal:
        experiment_name += "-gradrev"
    if dry_run:
        logger.info("Performing dry run to test pipeline!")
        experiment_name += "-dev"
    else:
        tracker = EnergyTracker(project_name=experiment_name)
        tracker.start()

    logger.start_run(run_name=experiment_name, tags={"model": model_name})

    if os.path.exists("sleepwalker.log"):
        os.remove("sleepwalker.log")

    logger.info(f"Loaded {train_dataset.get_n_patients()} for training")
    logger.info(f"Loaded {val_dataset.get_n_patients()} for validation")
    logger.info(f"Prepared {len(test_parts)} test dataset(s)")
    logger.info(f"Input data is {run_cfg['ts_len']} x {run_cfg['n_channels']}")
    summary(model, input_size=(1, run_cfg["ts_len"], run_cfg["n_channels"]), depth=5, row_settings=["hide_recursive_layers"])

    class_cnts = None
    if run_cfg["balance_batches"]:
        class_cnts = estimate_class_cnts(
            train_dataset,
            run_cfg["n_samples"],
            NUM_WORKERS_DATALOADER,
            run_cfg["batch_size"],
        )
        class_cnts_list = [class_cnts.get(c, 1) for c in train_dataset.get_classes()]
        callback = partial(
            prepare_multiclass_target,
            target_classes=train_dataset.get_classes(),
            class_cnts=class_cnts_list,
        )
        if isinstance(train_dataset, MultiDataset):
            for current_ds in train_dataset.datasets:
                current_ds.prepare_target_callback = callback
        else:
            train_dataset.prepare_target_callback = callback

    if run_cfg["loss_mode"] != "regular":
        if class_cnts is None:
            class_cnts = estimate_class_cnts(
                train_dataset,
                run_cfg["n_samples"],
                NUM_WORKERS_DATALOADER,
                run_cfg["batch_size"],
            )
        class_weights = class_weights_for_loss(run_cfg["class_weights"], class_cnts, run_cfg["loss_mode"])
    else:
        class_weights = run_cfg["class_weights"]

    if len(class_weights) > 0:
        weights_torch = torch.tensor([class_weights.get(c, 1) for c in train_dataset.get_classes()], device="cuda")
        trainer.loss_function = partial(run_cfg["loss_function"], weight=weights_torch)

    train_loader = build_loader(train_dataset, run_cfg["batch_size"], run_cfg["n_samples"], shuffle_default=True)
    val_loader = build_loader(val_dataset, run_cfg["batch_size"], run_cfg["n_samples"], shuffle_default=True)

    train_dict = trainer.fit(model, train_loader, val_loader)
    losses, cms = train_dict["losses"], train_dict["outputs"]
    if "checkpoint" in train_dict:
        with torch.inference_mode():
            state_dict = torch.load(os.path.join(train_dict["checkpoint"], "model.pt"), map_location="cpu")
            model.load_state_dict(state_dict)

    repeats = GROUPED_TEST_REPEATS if grouped else [1]
    for dataset_name, test_dataset in test_parts:
        for repeat in repeats:
            context_label = f"{dataset_name}:r={repeat}" if len(repeats) > 1 else dataset_name
            logger.context(context_label)
            if grouped:
                trainer.n_repeat_test = repeat

            test_loader = build_loader(test_dataset, run_cfg["batch_size"], None, shuffle_default=False)
            test_loss, test_cm = trainer.test(model, test_loader)
            record = {
                "model": f"{model_name}{'_GradReversal' if grad_reversal else ''}",
                "test_loss": test_loss,
                "test_cm": test_cm,
                "train_loss": losses,
                "train_cm": cms,
                "dataset": dataset_name,
                "classes": train_dataset.get_classes(),
                "artifact_root": train_dict["checkpoint"] if "checkpoint" in train_dict else "",
            }
            if len(repeats) > 1:
                record["repeat"] = repeat
            if "best_model" in train_dict:
                record["best_model"] = train_dict["best_model"]
            append_to_jsonl(experiment_name, record)
            logger.uncontext()

    if not dry_run:
        tracker.stop()
    logger.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a sleep-staging model.")
    parser.add_argument("--model", required=False, default="sleeptransformer", type=str)
    parser.add_argument("--train", required=False, nargs="+", default=["sleepedfx"], type=str)
    parser.add_argument("--test", required=False, nargs="*", default=None, type=str)
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--gradrev", action="store_true")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    run(args.model, args.train, args.test, args.dry, grouped=args.grouped, grad_reversal=args.gradrev)
