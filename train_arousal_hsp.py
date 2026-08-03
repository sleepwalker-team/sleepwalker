#!/bin/env python3

from __future__ import annotations

import numpy as np
np.seterr(all='raise')

import argparse
from collections import defaultdict
import os
from functools import partial

import pandas as pd
import yaml
from sleepwalker.models.preprocessors.RobustScaler import RobustScaler

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
from sleepwalker.datasets.utils import random_split
from sleepwalker.models import MultiModel, MetaModelEntry, SleepTransformer
from sleepwalker.models.UTime import UTime
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run, seed_everything
from sleepwalker.trainer.utils.filtering import trim_event
from sleepwalker.trainer.utils.splits import load_split
from sleepwalker.trainer.utils.targets import prepare_multiclass_target
from sleepwalker.utils import logger

try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
mp.set_sharing_strategy("file_system")

TARGET_CLASSES = ["no_arousal", "arousal"]

# FIXME: Remove these. They are not necessary
AROUSAL_CONFIG_KEYS = {
    "root", "id", "seed", "split", "log_path", "batch_size", "epochs", "n_samples",
    "num_workers_dataset", "num_workers_dataloader", "sample_frequency", "target_resolution",
    "stride", "grouped", "channel_groups", "robust_scaler", "arousal_weight", "balance_batches",
    "balance_gamma", "model", "total_input", "val_frac", "dry", "use_mlflow",
    "require_arousal_annotation", "max_train_patients", "max_val_patients", "max_test_patients",
    "test_window_limit", "allow_missing_channel_units", "loss", "save_every",
}
REQUIRED_AROUSAL_CONFIG_KEYS = {
    "root", "seed", "batch_size", "epochs", "n_samples", "num_workers_dataset",
    "num_workers_dataloader", "sample_frequency", "target_resolution", "stride", "grouped",
    "channel_groups", "robust_scaler", "arousal_weight", "model", "total_input", "dry",
    "use_mlflow", "loss", "save_every",
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
    dataset.initialize(patients, int(cfg["num_workers_dataset"]))
    return dataset

def cfg_to_channelcfg(cfg: dict) -> list[ChannelConfig]:
    return get_channels(
        cfg["channel_groups"],
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
        assume_units_if_missing=bool(cfg.get("allow_missing_channel_units", False)),
    )
    dataset.classes = list(TARGET_CLASSES)
    logger.info(
        f"Configured {len(dataset.get_input_channels())} effective input channels "
        f"from {len(channel_configs)} HSP channel candidates."
    )
    return dataset


def has_required_channels(edf_path: str, requested: list[ChannelConfig]) -> bool:
    meta = read_edf_meta(edf_path)
    available_channels = set(meta["signals"])

    channel_by_group = defaultdict(list)
    for channel_cfg in requested:
        if channel_cfg.group:
            channel_by_group[channel_cfg.group].append(channel_cfg.name)
        else:
            channel_by_group[channel_cfg.name].append(channel_cfg.name)

    for requested_channels in channel_by_group.values():
        if not any(channel in available_channels for channel in requested_channels):
            return False
    return True

def select_patients(records: list[str], channels: list[ChannelConfig], require_arousal_annotation: bool) -> list[str]:
    selected = []
    for record in records:
        if not has_required_channels(record, channels):
            continue
        annotation_path = get_hsp_annotation_path(record)
        if annotation_path is None:
            continue
        if require_arousal_annotation and int(get_hsp_annotation_label_counts(annotation_path).get("arousal", 0)) == 0:
            continue
        selected.append(record)
    return selected

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
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(7,n_channels)])] if not cfg["grouped"] else [RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(1, n_channels)])] if cfg["robust_scaler"] else None
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
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(7,n_channels)])] if not cfg["grouped"] else [RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(1, n_channels)])] if cfg["robust_scaler"] else None
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
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(7,n_channels)])] if not cfg["grouped"] else [RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(1, n_channels)])] if cfg["robust_scaler"] else None
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
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(0, n_channels-1)])] if cfg["robust_scaler"] else None
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
    
    if cfg["loss"] != "cross_entropy":
        raise ValueError(f"Unsupported loss {cfg['loss']!r}.")
    trainer = MulticlassTrainer(
        epochs=cfg["epochs"],
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ), 
        classes=TARGET_CLASSES,
        loss_function=torch.nn.functional.cross_entropy,
        save_every=int(cfg["save_every"]),
        # loss_mode="inverse",
        balance_batches=cfg.get("balance_batches", False),
        balance_gamma=cfg.get("balance_gamma", 0.75),
        class_weights={"no_arousal":1, "arousal":cfg["arousal_weight"]}
    )
    return model, trainer

def read_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    unknown = sorted(set(cfg) - AROUSAL_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown arousal config keys in {path}: {unknown}")
    missing = sorted(REQUIRED_AROUSAL_CONFIG_KEYS - set(cfg))
    if missing:
        raise ValueError(f"Missing arousal config keys in {path}: {missing}")
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an arousal model.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML run config.")
    parser.add_argument("--id", type=str, default=None, help="ID of the experiment")
    args = parser.parse_args()
    cfg = read_yaml_config(args.config)
    if args.id is not None:
        cfg["id"] = args.id
    seed_everything(int(cfg["seed"]))
    # cfg = normalize_config(cfg)

    experiment_name = "arousal" if cfg['id'] is None else f"arousal_{cfg['id']}" 
    if cfg["dry"]:
        logger.info("Performing dry run to test pipeline!")
        experiment_name += "-dev"
    
    log_path = str(cfg.get("log_path", os.path.join("results", "arousal")))
    os.makedirs(os.path.join(log_path, experiment_name), exist_ok=True)
    logger.set_log_file(os.path.join(log_path, experiment_name, "output.log"))

    requested_channels = cfg_to_channelcfg(cfg)
    require_arousal_annotation = bool(cfg.get("require_arousal_annotation", False))
    split_path = cfg.get("split")
    if split_path:
        split = load_split(split_path)
        task_split = split["tasks"]["arousal"]["edf_files"]
        train_patients = list(task_split["train"])
        val_patients = list(task_split["val"])
        test_patients = list(task_split["test"])
        for name, records in [ ("train", train_patients), ("val", val_patients), ("test", test_patients)]:
            limit = cfg.get(f"max_{name}_patients")
            if limit is not None:
                records[:] = records[: int(limit)]
        train_patients = select_patients(train_patients, requested_channels, require_arousal_annotation)
        val_patients = select_patients(val_patients, requested_channels, require_arousal_annotation)
        test_patients = select_patients(test_patients, requested_channels, require_arousal_annotation)
    else:
        train_patients = select_patients(
            get_annotated_hsp_edf_files(str(cfg["root"]), recursive=True),
            requested_channels,
            require_arousal_annotation,
        )
        if cfg.get("val_frac", 0.1) > 0:
            train_patients, val_patients = random_split(
                train_patients,
                test_frac=cfg.get("val_frac", 0.1),
                seed=int(cfg["seed"]),
            )
        else:
            val_patients = []
        test_patients = []
    val_dataset = None
    logger.context("TRAIN")
    train_dataset = build_dataset(train_patients, cfg)
    logger.uncontext()
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
    run(
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
            n_samples_test=cfg.get("test_window_limit"),
            test_repeats=[1,2] if cfg["grouped"] else [1],
            tags={"model": cfg["model"]},
            collate_fn=batch_collate,
            use_mlflow=cfg["use_mlflow"],
            log_path=log_path,
            meta_data=cfg,
            expert_task="arousal",
        )
    )


if __name__ == "__main__":
    main()
