#!/usr/bin/env python3
"""Train HSP dense event experts for breathing and desaturation tasks."""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import partial
import multiprocessing
import os
from typing import Optional

import torch
import torch.multiprocessing as mp
import yaml

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.HSP import HSP, get_annotated_hsp_edf_files, get_channels, get_hsp_annotation_path
from sleepwalker.datasets.utils import get_edf_files_in_repo, random_split
from sleepwalker.models.UTime import UTime
from sleepwalker.models.preprocessors.RobustScaler import RobustScaler
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.utils.filtering import trim_event
from sleepwalker.trainer.utils.targets import prepare_multiclass_target
from sleepwalker.utils import logger, suppress_stdout_logging

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
mp.set_sharing_strategy("file_system")

TASKS = {
    "breathing": {
        "classes": ["apnea", "hypopnea", "regular breathing"],
        "channels": ["abdomen", "chest", "airflow", "spo2"],
        "target_resolution": "5s",
        "stride": "1s",
        "total_input": "60s",
        "event_mapping": {
            "n1": "sleep",
            "n2": "sleep",
            "n3": "sleep",
            "rem": "sleep",
            "apnea": "apnea",
            "obstructive-apnea": "apnea",
            "central-apnea": "apnea",
            "mixed-apnea": "apnea",
            "hypopnea": "hypopnea",
        },
        "class_weights": {"apnea": 2.0, "hypopnea": 2.0, "regular breathing": 1.0},
    },
    "desaturation": {
        "classes": ["desaturation", "no desaturation"],
        "channels": ["spo2"],
        "target_resolution": "10s",
        "stride": "2s",
        "total_input": "100s",
        "event_mapping": {
            "n1": "sleep",
            "n2": "sleep",
            "n3": "sleep",
            "rem": "sleep",
            "desaturation": "desaturation",
        },
        "class_weights": {"desaturation": 2.0, "no desaturation": 1.0},
    },
}

DEFAULT_CONFIG = {
    "root": "/raid/sleepwalker/hsp",
    "task": "desaturation",
    "batch_size": 128,
    "epochs": 2,
    "n_samples": 10000,
    "num_workers_dataset": 2,
    "num_workers_dataloader": 8,
    "sample_frequency": 100,
    "grouped": True,
    "scaler": True,
    "model": "utime-big",
    "sleep_percentage": 0.5,
    "val_frac": 0.1,
    "test_frac": 0.1,
    "dry": False,
    "use_mlflow": True,
    "max_edf_files": None,
    "patient_limit": None,
    "annotated_only": True,
    "balance_batches": False,
    "balance_gamma": 0.75,
}


def read_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(loaded)
    task_defaults = TASKS[cfg["task"]]
    for key in ["channels", "target_resolution", "stride", "total_input"]:
        cfg.setdefault(key, task_defaults[key])
    cfg.setdefault("class_weights", task_defaults["class_weights"])
    return cfg


def prepare_patient(data_df, label_df, label_extra_df, patient=None):
    trimmed = trim_event(data_df, label_df, label_extra_df, ["sleep"])
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df


def prepare_sample(data, quality_data=None, target=None, target_extra=None, patient=None, time=None):
    if float(data.isna().mean().mean()) > 0.05:
        return None

    for col in ["SpO2", "Saturation", "SaO2"]:
        if col in data.columns:
            invalid_spo2_fraction = float(((data[col] < -5.0) | (data[col] > 5.0)).mean())
            if invalid_spo2_fraction > 0.05:
                return None

    item = {
        "data": torch.from_numpy(data.values).float(),
        "target": target,
        "patient": patient,
        "time": time,
    }
    if target_extra is not None:
        item["target_extra"] = target_extra
    return item


def cfg_to_channelcfg(cfg: dict):
    return get_channels(
        cfg["channels"],
        grouped=bool(cfg["grouped"]),
        normalize=True,
        sample_frequency=float(cfg["sample_frequency"]),
        override_normalize=None,
    )


def build_dataset_template(cfg: dict):
    task_cfg = TASKS[cfg["task"]]
    prepare_target = partial(
        prepare_multiclass_target,
        target_classes=task_cfg["classes"],
        filters=[{"columns": ["sleep"], "percentage": float(cfg["sleep_percentage"]), "mode": "min"}],
    )

    dataset = HSP(
        channels=cfg_to_channelcfg(cfg),
        sample_frequency=float(cfg["sample_frequency"]),
        event_mapping=task_cfg["event_mapping"],
        stride=str(cfg["stride"]),
        prepare_patient=prepare_patient,
        prepare_target=prepare_target,
        prepare_sample=prepare_sample,
        total_input=str(cfg["total_input"]),
        target_resolution=str(cfg["target_resolution"]),
    )
    dataset.classes = list(task_cfg["classes"])
    logger.info(
        f"Configured {len(dataset.get_input_channels())} effective input channels "
        f"from {len(dataset.channels)} HSP channel candidates."
    )
    return dataset


def build_dataset(patients: list[str], cfg: dict):
    dataset = build_dataset_template(cfg)
    dataset.initialize(patients, int(cfg["num_workers_dataset"]))
    return dataset


def is_usable(edf_path: str, cfg: dict) -> Optional[str]:
    if get_hsp_annotation_path(edf_path) is None:
        return None

    meta = read_edf_meta(edf_path)
    available_channels = set(meta["signals"])
    requested = cfg_to_channelcfg(cfg)

    channel_by_group = defaultdict(list)
    for channel_cfg in requested:
        key = channel_cfg.group or channel_cfg.name
        channel_by_group[key].append(channel_cfg.name)

    for requested_channels in channel_by_group.values():
        if not any(channel in available_channels for channel in requested_channels):
            return None
    return edf_path


def list_patients(cfg: dict) -> list[str]:
    if cfg.get("annotated_only", True):
        edf_files = get_annotated_hsp_edf_files(cfg["root"], recursive=True)
    else:
        edf_files = get_edf_files_in_repo(cfg["root"], recursive=True)
    if cfg.get("max_edf_files") is not None:
        edf_files = edf_files[: int(cfg["max_edf_files"])]

    logger.progress_start(len(edf_files), desc="Collecting patients", leave=True)
    patients = []
    if int(cfg["num_workers_dataset"]) > 1:
        with multiprocessing.Pool(int(cfg["num_workers_dataset"])) as pool:
            iter_objects = pool.imap_unordered(partial(is_usable, cfg=cfg), edf_files)
            for result in iter_objects:
                if result:
                    patients.append(result)
                logger.progress_advance(1)
    else:
        for patient in edf_files:
            if is_usable(patient, cfg) is not None:
                patients.append(patient)
            logger.progress_advance(1)

    logger.progress_close()
    logger.info(f"Collected patient stats for {len(patients)}/{len(edf_files)} patients.")
    if cfg.get("patient_limit") is not None:
        patients = patients[: int(cfg["patient_limit"])]
    return patients


def build_model_and_trainer(train_dataset, cfg: dict):
    n_channels = len(train_dataset.get_input_channels())
    preprocessors = (
        [RobustScaler(lower_quantile=0.1, upper_quantile=0.9, channels=list(range(n_channels)))]
        if bool(cfg.get("scaler", False))
        else None
    )
    task_cfg = TASKS[cfg["task"]]

    if cfg["model"] == "utime-big":
        model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=n_channels,
            classes=task_cfg["classes"],
            sampling_frequency=float(cfg["sample_frequency"]),
            channel=[64, 128, 128, 256],
            kernel=[5, 5, 3, 3],
            maxpool=[5, 5, 3, 3],
            norm="channel",
            activation="elu",
            mlp_size=512,
            dropout_p=0,
            preprocessors=preprocessors,
        )
    elif cfg["model"] == "utime-small":
        model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=n_channels,
            classes=task_cfg["classes"],
            sampling_frequency=float(cfg["sample_frequency"]),
            channel=[64, 128, 256],
            kernel=[5, 3, 3],
            maxpool=[5, 3, 3],
            norm="channel",
            activation="elu",
            mlp_size=64,
            dropout_p=0,
            preprocessors=preprocessors,
        )
    else:
        raise ValueError(f"Did not recognize model {cfg['model']}")

    trainer = MulticlassTrainer(
        epochs=2 if bool(cfg.get("dry", False)) else int(cfg["epochs"]),
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ),
        classes=task_cfg["classes"],
        loss_function=torch.nn.functional.cross_entropy,
        save_every=1 if bool(cfg.get("dry", False)) else 10,
        balance_batches=bool(cfg.get("balance_batches", False)),
        balance_gamma=float(cfg.get("balance_gamma", 0.75)),
        class_weights=dict(cfg.get("class_weights", task_cfg["class_weights"])),
    )
    return model, trainer


def build_expert_components(cfg: dict):
    resolved = dict(DEFAULT_CONFIG)
    resolved.update(cfg)
    task_defaults = TASKS[resolved["task"]]
    for key in ["channels", "target_resolution", "stride", "total_input"]:
        resolved.setdefault(key, task_defaults[key])
    dataset = build_dataset_template(resolved)
    model, trainer = build_model_and_trainer(dataset, resolved)
    return {
        "model": model,
        "trainer": trainer,
        "dataset_template": dataset.to_unlabelled(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="YAML config path.")
    parser.add_argument("--id", type=str, default=None, help="Optional experiment id override.")
    args = parser.parse_args()

    cfg = read_yaml_config(args.config)
    if args.id is not None:
        cfg["id"] = args.id
    cfg.setdefault("id", None)

    experiment_name = cfg["task"] if cfg["id"] is None else f"{cfg['task']}_{cfg['id']}"
    if bool(cfg.get("dry", False)):
        logger.info("Performing dry run to test pipeline!")
        experiment_name += "-dev"

    log_dir = os.path.join("results", cfg["task"], experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    logger.set_log_file(os.path.join(log_dir, "output.log"))

    patients = list_patients(cfg)
    if len(patients) == 0:
        raise ValueError("No usable HSP patients found.")

    if float(cfg.get("test_frac", 0.0)) > 0:
        train_patients, test_patients = random_split(patients, test_frac=float(cfg["test_frac"]))
    else:
        train_patients, test_patients = list(patients), []
    if float(cfg.get("val_frac", 0.0)) > 0:
        train_patients, val_patients = random_split(train_patients, test_frac=float(cfg["val_frac"]))
    else:
        val_patients = []

    with suppress_stdout_logging(logger):
        logger.context("TRAIN")
        train_dataset = build_dataset(train_patients, cfg)
        logger.uncontext()
        val_dataset = None
        if len(val_patients) > 0:
            logger.context("VAL")
            val_dataset = build_dataset(val_patients, cfg)
            logger.uncontext()
        test_dataset = None
        if len(test_patients) > 0:
            logger.context("TEST")
            test_dataset = build_dataset(test_patients, cfg)
            logger.uncontext()

    model, trainer = build_model_and_trainer(train_dataset, cfg)
    expert_builder_config = {
        "task": cfg["task"],
        "channels": list(cfg["channels"]),
        "sample_frequency": cfg["sample_frequency"],
        "stride": cfg["stride"],
        "total_input": cfg["total_input"],
        "target_resolution": cfg["target_resolution"],
        "sleep_percentage": cfg["sleep_percentage"],
        "grouped": cfg["grouped"],
        "model": cfg["model"],
        "scaler": cfg["scaler"],
        "class_weights": cfg.get("class_weights", TASKS[cfg["task"]]["class_weights"]),
        "epochs": 2 if bool(cfg.get("dry", False)) else int(cfg["epochs"]),
    }
    expert_dataset_template = build_expert_components(expert_builder_config)["dataset_template"]

    run(
        RunCfg(
            experiment_name=experiment_name,
            model_name=cfg["model"],
            model=model,
            trainer=trainer,
            train_datasets=[train_dataset],
            val_datasets=[] if val_dataset is None else [val_dataset],
            test_datasets=[] if test_dataset is None else [("HSP", test_dataset)],
            batch_size=int(cfg["batch_size"]),
            n_samples=1_000 if bool(cfg.get("dry", False)) else int(cfg["n_samples"]),
            num_workers_dataloader=int(cfg["num_workers_dataloader"]),
            test_repeats=[1],
            use_energy_tracker=False,
            tags={"model": cfg["model"], "task": cfg["task"], "dataset": "hsp"},
            collate_fn=batch_collate,
            use_mlflow=bool(cfg.get("use_mlflow", True)),
            log_path=os.path.join("results", cfg["task"]),
            meta_data=cfg,
            expert_name=experiment_name,
            expert_task=cfg["task"],
            expert_builder={
                "module": "train_event_hsp",
                "function": "build_expert_components",
                "config": expert_builder_config,
            },
            expert_dataset_template=expert_dataset_template,
        )
    )


if __name__ == "__main__":
    main()
