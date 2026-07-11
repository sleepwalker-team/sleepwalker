#!/bin/env python3

from __future__ import annotations

import numpy as np
np.seterr(all='raise')

import argparse
from collections import defaultdict
import multiprocessing
import os
from functools import partial
from typing import Optional

import pandas as pd
import yaml
from sleepwalker.models.preprocessors.RobustScaler import RobustScaler
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import torch
import torch.multiprocessing as mp

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets.HSP import (
    HSP,
    get_annotated_hsp_edf_files,
    get_channels,
    get_hsp_annotation_label_counts,
    get_hsp_annotation_path,
)
from sleepwalker.datasets.Basedataset import ChannelConfig, batch_collate
from sleepwalker.datasets.utils import get_edf_files_in_repo, random_split
from sleepwalker.models import MultiModel, MetaModelEntry, SleepTransformer
from sleepwalker.models.UTime import UTime
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.utils.filtering import trim_event
from sleepwalker.trainer.utils.targets import prepare_multiclass_target
from sleepwalker.utils import logger, suppress_stdout_logging

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
mp.set_sharing_strategy("file_system")

ROOT = "/cephfs_projects/sleepwalker/hsp"

TARGET_CLASSES = ["no_arousal", "arousal"]
DEFAULT_CONFIG = {
    "root": ROOT,
    "task": "arousal",
    "batch_size": 128,
    "epochs": 35,
    "n_samples": 100_000,
    "num_workers_dataset": 8,
    "num_workers_dataloader": 8,
    "sample_frequency": 100,
    "target_resolution": "1s",
    "stride": "1s",
    "grouped": False,
    "channels": ["eeg", "eog", "chin_emg"],
    "scaler": False,
    "arousal_weight": 1,
    "balance_batches": False,
    "balance_gamma": 0.75,
    "model": "utime-big",
    "id": None,
    "total_input": "60s",
    "val_frac": 0.1,
    "dry": False,
    "use_mlflow": True,
    "max_edf_files": None,
    "patient_limit": None,
    "annotated_only": True,
    "require_positive_labels": True,
}


def infer_sleeptransformer_epoch_seq_len(total_input: str) -> int:
    total_input_seconds = pd.to_timedelta(total_input).total_seconds()
    approx_seq_len = total_input_seconds / 21
    epoch_seq_len = max(1, int(round(approx_seq_len)))
    if epoch_seq_len % 2 == 0:
        epoch_seq_len += 1 if approx_seq_len >= epoch_seq_len else -1
    # Keep the 630s -> 29 SleepTransformer anchor while preserving an odd center token.
    return min(epoch_seq_len, 29)

def prepare_patient(data_df, label_df, label_extra_df, patient=None):
    trimmed = trim_event(data_df, label_df, label_extra_df, ["sleep"])
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df

def prepare_sample(
    data,
    target=None,
    target_extra=None,
    patient=None,
    quality_data=None,
    time=None,
):
    # if quality_data is not None and "EEG" in quality_data.columns:
    #     if float(quality_data["EEG"].astype(float).mean()) > impedance_cutoff_ohm:
    #         return None

    # if float(data.isna().mean().mean()) > 0.05:
    #     return None

    # for col in ["EEG", "C3-M2", "ECG", "Pulse Waveform"]:
    #     if col in data.columns and float(data[col].std()) < 1e-3:
    #         return None

    item = {
        "data": torch.from_numpy(data.values).float(),
        "target": target,
        "patient": patient,
        "time": time,
    }
    if target_extra is not None:
        item["target_extra"] = target_extra
    return item

def build_dataset(patients: list[str], cfg: dict):
    dataset = build_dataset_template(cfg)
    dataset.initialize(patients, cfg["num_workers_dataset"])
    return dataset

def cfg_to_channelcfg(cfg: dict) -> list[ChannelConfig]:
    return get_channels(
        cfg["channels"],
        grouped=cfg["grouped"],
        normalize=True,
        sample_frequency=cfg["sample_frequency"],
        override_normalize={"chin_emg":None, "leg_emg": None, "EKG":None},
    )

def build_dataset_template(cfg: dict):
    EVENT_MAPPING = {
        "arousal": "arousal",
        "n1": "sleep",
        "n2": "sleep",
        "n3": "sleep",
        "rem": "sleep",
        #"rera": "arousal", 
        #"plm-arousal": "arousal", 
    }

    prepare_arousal_target = partial(
        prepare_multiclass_target,
        target_classes=TARGET_CLASSES,
    )

    channel_configs = cfg_to_channelcfg(cfg)
    dataset = HSP(
        channels=channel_configs,
        sample_frequency=cfg["sample_frequency"],
        event_mapping=EVENT_MAPPING,
        stride=cfg["stride"],
        prepare_patient=prepare_patient,
        prepare_target=prepare_arousal_target,
        prepare_sample=prepare_sample,
        total_input=cfg["total_input"],
        target_resolution=cfg["target_resolution"],
    )
    dataset.classes = list(TARGET_CLASSES)
    logger.info(
        f"Configured {len(dataset.get_input_channels())} effective input channels "
        f"from {len(channel_configs)} HSP channel candidates."
    )
    return dataset


def is_usable(edf_path: str, cfg: dict) -> Optional[str]:
    annot_path = get_hsp_annotation_path(edf_path)
    if annot_path is None:
        return None
    if cfg.get("require_positive_labels", True):
        label_counts = get_hsp_annotation_label_counts(annot_path)
        if label_counts.get("arousal", 0) <= 0:
            return None

    meta = read_edf_meta(edf_path)
    available_channels = set(meta["signals"])
    requested = cfg_to_channelcfg(cfg)

    channel_by_group = defaultdict(list)
    for channel_cfg in requested:
        if channel_cfg.group:
            channel_by_group[channel_cfg.group].append(channel_cfg.name)
        else:
            channel_by_group[channel_cfg.name].append(channel_cfg.name)

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
    if cfg["num_workers_dataset"] > 1:
        with multiprocessing.Pool(cfg["num_workers_dataset"]) as pool:
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
    model_name = cfg["model"]
    if model_name == "utime-big":
        model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=n_channels,
            classes=TARGET_CLASSES, #["arousal"],
            sampling_frequency=cfg["sample_frequency"],
            channel = [64, 128, 128, 256],
            kernel = [5, 5, 3, 3],
            maxpool = [5, 5, 3, 3],
            norm="channel",
            activation="elu",
            mlp_size=512,
            dropout_p=0,
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(7,n_channels)])] if not cfg["grouped"] else [RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(1, n_channels)])] if cfg["scaler"] else None
        )
    elif model_name == "utime-huge":
        model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=n_channels,
            classes=TARGET_CLASSES, #["arousal"],
            sampling_frequency=cfg["sample_frequency"],
            channel = [32, 64, 128, 128, 256],
            kernel = [5, 5, 3, 3, 3],
            maxpool = [5, 5, 3, 3, 3],
            norm="channel",
            activation="elu",
            mlp_size=1024,
            dropout_p=0,
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(7,n_channels)])] if not cfg["grouped"] else [RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(1, n_channels)])] if cfg["scaler"] else None
        )
    elif model_name == "utime-small":
        model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=n_channels,
            classes=TARGET_CLASSES, #["arousal"],
            sampling_frequency=cfg["sample_frequency"],
            channel = [64, 128, 256],
            kernel = [5, 3, 3],
            maxpool = [5, 3, 3],
            norm="channel",
            activation="elu",
            mlp_size=64,
            dropout_p=0,
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(7,n_channels)])] if not cfg["grouped"] else [RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(1, n_channels)])] if cfg["scaler"] else None 
        )
    elif model_name == "multi":
        if cfg["grouped"]:
            eeg_channels = ["EEG"]
        else:
            eeg_channels = ["C3-M2", "C4-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"]
        
        aux_channels = [channel for channel in train_dataset.get_input_channels() if channel not in eeg_channels]
        if len(eeg_channels) == 0 or len(aux_channels) == 0:
            raise ValueError("Multimodel requires at least one EEG and one non-EEG input channel.")

        aux_model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=len(aux_channels),
            classes=None,
            sampling_frequency=cfg["sample_frequency"],
            channel=[32, 64, 128],
            maxpool=[8, 6, 4],
            kernel=[5, 3, 3],
            norm="channel",
            mlp_size=64,
            dropout_p=0,
            activation="elu",
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(0, n_channels-1)])] if cfg["scaler"] else None 
        )
        eeg_model = SleepTransformer(
            classes=None,
            n_channels=len(eeg_channels),
            epoch_seq_len=infer_sleeptransformer_epoch_seq_len(cfg["total_input"]),
        )
        model = MultiModel(
            classes=TARGET_CLASSES,
            input_channels=train_dataset.get_input_channels(),
            models=[
                MetaModelEntry(aux_model, aux_channels),
                MetaModelEntry(eeg_model, eeg_channels),
            ],
        )
    else:
        raise ValueError(f"Did not recoginize model {model_name}")
    
    trainer = MulticlassTrainer(
        epochs=cfg["epochs"],
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ), 
        classes=TARGET_CLASSES,
        loss_function=torch.nn.functional.cross_entropy,#torch.nn.functional.binary_cross_entropy_with_logits,
        save_every=10,
        # loss_mode="inverse",
        balance_batches=cfg.get("balance_batches", False),
        balance_gamma=cfg.get("balance_gamma", 0.75),
        class_weights={"no_arousal":1, "arousal":cfg["arousal_weight"]}
    )
    return model, trainer

def build_expert_components(cfg: dict):
    cfg = dict(cfg)
    dataset = build_dataset_template(cfg)
    model, trainer = build_model_and_trainer(dataset, cfg)
    return {
        "model": model,
        "trainer": trainer,
        "dataset_template": dataset.to_unlabelled(),
    }

def read_yaml_config(path: str) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as handle:
        loaded_cfg = yaml.safe_load(handle) or {}
    if not isinstance(loaded_cfg, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    cfg.update(loaded_cfg)
    return cfg

def log_patient_split(train_patients, val_patients):
    with open('train_hsp.txt', 'w') as f:
        for line in train_patients:
            f.write(f"{line}\n")
    with open('val_hsp.txt', 'w') as f:
        for line in val_patients:
            f.write(f"{line}\n")
    logger.artifact("train_hsp.txt", "train.txt")
    logger.artifact("val_hsp.txt", "val.txt")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an arousal model.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML run config.")
    parser.add_argument("--id", type=str, default=None, help="ID of the experiment")
    args = parser.parse_args()
    cfg = read_yaml_config(args.config)
    if args.id is not None:
        cfg["id"] = args.id
    # cfg = normalize_config(cfg)

    experiment_name = "arousal" if cfg['id'] is None else f"arousal_{cfg['id']}" 
    if cfg["dry"]:
        logger.info("Performing dry run to test pipeline!")
        experiment_name += "-dev"
    
    os.makedirs(os.path.join("results", "arousal", experiment_name), exist_ok=True)
    logger.set_log_file(os.path.join("results", "arousal", experiment_name, "output.log"))

    train_patients = list_patients(cfg)
    val_dataset = None
    if cfg["val_frac"] > 0:
        train_patients, val_patients = random_split(train_patients, test_frac=cfg["val_frac"],seed=1)
    else:
        val_patients = []
    log_patient_split(train_patients, val_patients)

    # with suppress_stdout_logging(logger):
    logger.context("TRAIN")
    train_dataset = build_dataset(train_patients, cfg)
    logger.uncontext()
    if len(val_patients) > 0:
        logger.context("VAL")
        val_dataset = build_dataset(val_patients, cfg)
        logger.uncontext()
    test_dataset = None 

    trainer_cfg = dict(cfg)
    trainer_cfg["epochs"] = 2 if cfg["dry"] else cfg["epochs"]
    model, trainer = build_model_and_trainer(train_dataset, trainer_cfg)
    expert_builder_config = {
        "sample_frequency": cfg["sample_frequency"],
        "task": cfg.get("task", "arousal"),
        "target_resolution": cfg["target_resolution"],
        "stride": cfg["stride"],
        "channels": list(cfg["channels"]),
        "clean": cfg["clean"],
        "grouped": cfg["grouped"],
        "total_input": cfg["total_input"],
        "model": cfg["model"],
        "scaler": cfg["scaler"],
        "arousal_weight": cfg["arousal_weight"],
        "balance_batches": cfg.get("balance_batches", False),
        "balance_gamma": cfg.get("balance_gamma", 0.75),
        "epochs": trainer_cfg["epochs"],
        "num_workers_dataset": cfg["num_workers_dataset"],
        "max_edf_files": cfg.get("max_edf_files"),
        "patient_limit": cfg.get("patient_limit"),
        "annotated_only": cfg.get("annotated_only", True),
        "require_positive_labels": cfg.get("require_positive_labels", True),
    }
    expert_dataset_template = build_expert_components(expert_builder_config)["dataset_template"]

    run_result = run(
        RunCfg(
            experiment_name=experiment_name,
            model_name=cfg["model"],
            model=model,
            trainer=trainer,
            train_datasets=[train_dataset],
            val_datasets=[] if val_dataset is None else [val_dataset],
            test_datasets=[] if test_dataset is None else [("HSPTest", test_dataset)],
            batch_size=cfg["batch_size"],
            n_samples=cfg["n_samples"],
            num_workers_dataloader=cfg["num_workers_dataloader"],
            test_repeats=[1,2] if cfg["grouped"] else [1],
            use_energy_tracker=False,
            tags={"model": cfg["model"]},
            collate_fn=batch_collate,
            use_mlflow=cfg["use_mlflow"],
            log_path=os.path.join("results", "arousal"),
            meta_data={**cfg, "config": args.config},
            expert_name=experiment_name,
            expert_task="arousal",
            expert_builder={
                "module": "train_arousal_hsp",
                "function": "build_expert_components",
                "config": expert_builder_config,
            },
            expert_dataset_template=expert_dataset_template,
        )
    )


if __name__ == "__main__":
    main()
