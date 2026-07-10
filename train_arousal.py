#!/bin/env python3

from __future__ import annotations

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
from sleepwalker.datasets import ChannelConfig, Ruhrlandklinik
from sleepwalker.datasets.Ruhrlandklinik import get_channels
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.normalizer.PulseFilterNormalizer import PulseFilterNormalizer
from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer
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

TRAIN_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/train-test-2023"
TEST_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/val-2024"

TARGET_CLASSES = ["no_arousal", "arousal"]
DEFAULT_CONFIG = {
    "train_root": TRAIN_ROOT,
    "test_root": TEST_ROOT,
    "batch_size": 128,
    "epochs": 35,
    "n_samples": 100_000,
    "num_workers_dataset": 8,
    "num_workers_dataloader": 8,
    "sample_frequency": 100,
    "target_resolution": "1s",
    "stride": "1s",
    "grouped": False,
    "channels": ["eeg", "eog", "chin_emg", "ECG"],
    "scaler": False,
    "clean": False,
    "arousal_weight": 1,
    "model": "utime-big",
    "id": "",
    "total_input": "30s",
    "val_frac": 0.1,
    "dry": False,
    "use_mlflow": True,
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
    quality_data=None,
    target=None,
    target_extra=None,
    patient=None,
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

    if cfg["clean"]:
        EVENT_MAPPING["artefakt"] = "artifact"
        prepare_arousal_target = partial(
            prepare_multiclass_target,
            target_classes=TARGET_CLASSES,
            filters=[
                {"columns": ["wake"], "percentage": 0.5, "mode": "max"},
                {"columns": ["artifact"], "percentage": 0.0, "mode": "max"},
            ],
        )
    else:
        prepare_arousal_target = partial(
            prepare_multiclass_target,
            target_classes=TARGET_CLASSES,
        )

    channel_configs = get_channels(
        cfg["channels"],
        grouped=cfg["grouped"],
        include_quality=False,
        normalize=True,
        sample_frequency=cfg["sample_frequency"],
        override_normalize={"chin_emg": None, "ECG": None},
    )
    dataset = Ruhrlandklinik(
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
        f"from {len(channel_configs)} Ruhrland channel candidates."
    )
    return dataset


def build_dataset(patients: list[str], cfg: dict):
    dataset = build_dataset_template(cfg)
    dataset.initialize(patients, cfg["num_workers_dataset"])
    return dataset

def is_pap_patient(edf_path: str) -> bool:
    meta = read_edf_meta(edf_path)
    available_channels = meta["signals"]
    pap_channel_patterns = [
        "Druckeinstellung",
        "EPAP",
        "IPAP",
        "Druck (PAP)",
        "Mask Pressure",
        "Leck (PAP)",
        "Fluss (PAP)",
        "SpO2 (PAP)",
        "Puls (PAP)",
        "FiO2 (PAP)",
        "PrismaLeak",
        "PrismaFlow",
        "AutoPressure",
        "Ventilation Vorg",
        "AchievedAlveolar",
    ]
    return any(ch in available_channels for ch in pap_channel_patterns)


def has_required_channels(edf_path: str, cfg: dict) -> bool:
    meta = read_edf_meta(edf_path)
    available_channels = set(meta["signals"])
    requested = get_channels(
        cfg["channels"],
        grouped=cfg["grouped"],
        include_quality=False,
        normalize=True,
        sample_frequency=cfg["sample_frequency"],
        override_normalize={"chin_emg":None, "ECG":None, "Pulse Waveform":None},
    ) # TODO this has to be set in two places now
    channel_by_group = defaultdict(list)
    for cfg in requested:
        if cfg.group:
            channel_by_group[cfg.group].append(cfg.name)
        else:
            channel_by_group[cfg.name].append(cfg.name)

    for requested_channels in channel_by_group.values():
        if not any(channel in available_channels for channel in requested_channels):
            return False
    return True

def list_patients(source_root: str, cfg: dict) -> list[str]:
    patients = [
        patient for patient in get_edf_files_in_repo(source_root, recursive=True) if not is_pap_patient(patient) and has_required_channels(patient, cfg)
    ]
    return patients[:2] if cfg["dry"] else patients

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
        balance_batches=True,
        balance_gamma=0.75,
        class_weights={"no_arousal":1, "arousal":cfg["arousal_weight"]}
    )
    return model, trainer


def build_expert_components(cfg: dict):
    cfg = {**DEFAULT_CONFIG, **dict(cfg)}
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
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    cfg.update(loaded_cfg)
    return cfg

# def normalize_config(raw_cfg: dict) -> dict:
#     cfg = dict(DEFAULT_CONFIG)
#     cfg.update(raw_cfg)
#     cfg["train_root"] = str(cfg["train_root"])
#     cfg["test_root"] = str(cfg["test_root"])
#     cfg["batch_size"] = int(cfg["batch_size"])
#     cfg["epochs"] = int(cfg["epochs"])
#     cfg["n_samples"] = int(cfg["n_samples"])
#     cfg["num_workers_dataset"] = int(cfg["num_workers_dataset"])
#     cfg["num_workers_dataloader"] = int(cfg["num_workers_dataloader"])
#     cfg["sample_frequency"] = int(cfg["sample_frequency"])
#     cfg["target_resolution"] = str(cfg["target_resolution"])
#     cfg["stride"] = str(cfg["stride"])
#     cfg["channels"] = list(cfg["channels"])
#     cfg["grouped"] = bool(cfg["grouped"])
#     cfg["scaler"] = bool(cfg["scaler"])
#     cfg["clean"] = bool(cfg["clean"])
#     cfg["arousal_weight"] = int(cfg["arousal_weight"])
#     cfg["model"] = str(cfg["model"])
#     cfg["id"] = str(cfg["id"])
#     cfg["total_input"] = str(cfg["total_input"])
#     cfg["val_frac"] = float(cfg["val_frac"])
#     cfg["dry"] = bool(cfg["dry"])
#     cfg["use_mlflow"] = bool(cfg["use_mlflow"])
#     return cfg


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

    train_patients = list_patients(cfg["train_root"], cfg)
    test_patients = list_patients(cfg["test_root"], cfg)

    val_dataset = None
    if cfg["val_frac"] > 0:
        train_patients, val_patients = random_split(train_patients, test_frac=cfg["val_frac"])
    else:
        val_patients = []

    # with suppress_stdout_logging(logger):
    logger.context("TRAIN")
    train_dataset = build_dataset(train_patients, cfg)
    logger.uncontext()
    if len(val_patients) > 0:
        logger.context("VAL")
        val_dataset = build_dataset(val_patients, cfg)
        logger.uncontext()
    logger.context("TEST")
    test_dataset = build_dataset(test_patients, cfg)
    logger.uncontext()

    trainer_cfg = dict(cfg)
    trainer_cfg["epochs"] = 2 if cfg["dry"] else cfg["epochs"]
    model, trainer = build_model_and_trainer(train_dataset, trainer_cfg)
    expert_builder_config = {
        "sample_frequency": cfg["sample_frequency"],
        "target_resolution": cfg["target_resolution"],
        "stride": cfg["stride"],
        "channels": list(cfg["channels"]),
        "clean": cfg["clean"],
        "grouped": cfg["grouped"],
        "total_input": cfg["total_input"],
        "model": cfg["model"],
        "scaler": cfg["scaler"],
        "arousal_weight": cfg["arousal_weight"],
        "epochs": trainer_cfg["epochs"],
        "num_workers_dataset": cfg["num_workers_dataset"],
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
            test_datasets=[("Ruhrland2024", test_dataset)],
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
                "module": "train_arousal",
                "function": "build_expert_components",
                "config": expert_builder_config,
            },
            expert_dataset_template=expert_dataset_template,
        )
    )


if __name__ == "__main__":
    main()
