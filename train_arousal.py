#!/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
import os
from functools import partial

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
from sleepwalker.deployment import export_prediction_package
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.utils.filtering import trim_wake
from sleepwalker.trainer.utils.targets import prepare_multiclass_target
from sleepwalker.utils import logger, suppress_stdout_logging

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
mp.set_sharing_strategy("file_system")

TRAIN_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/train-test-2023"
TEST_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/val-2024"
BATCH_SIZE = 128
EPOCHS = 35
N_SAMPLES = 100_000
NUM_WORKERS_DATASET = 8
NUM_WORKERS_DATALOADER = 8
SAMPLE_FREQUENCY = 100
TOTAL_INPUT = "30s" #120s
TARGET_RESOLUTION = "1s" #0.5s
STRIDE = "1s"            #0.5s
SLEEP_PERCENTAGE = 0.5
TARGET_CLASSES = ["no_arousal", "arousal"]

def prepare_patient(data_df, label_df, label_extra_df, patient=None):
    trimmed = trim_wake(data_df, label_df, label_extra_df)
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

def build_dataset(patients: list[str], channels: list[str], clean:bool, grouped: bool):
    EVENT_MAPPING = {
        "arousal": "arousal",
        #"rera": "arousal", 
        #"plm-arousal": "arousal", 
    }

    if clean:
        EVENT_MAPPING["wach"] = "wake"
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
        channels,
        grouped=grouped,
        include_quality=False,
        normalize=True,
        sample_frequency=SAMPLE_FREQUENCY,
        override_normalize={"chin_emg": None, "ECG": None},
    )
    dataset = Ruhrlandklinik(
        channels=channel_configs,
        sample_frequency=SAMPLE_FREQUENCY,
        event_mapping=EVENT_MAPPING,
        stride=STRIDE,
        prepare_patient=prepare_patient,
        prepare_target=prepare_arousal_target,
        prepare_sample=prepare_sample,
        total_input=TOTAL_INPUT,
        target_resolution=TARGET_RESOLUTION,
    )
    dataset.classes = list(TARGET_CLASSES)
    logger.info(
        f"Configured {len(dataset.get_input_channels())} effective input channels "
        f"from {len(channel_configs)} Ruhrland channel candidates."
    )
    dataset.initialize(patients, NUM_WORKERS_DATASET)
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


def has_required_channels(edf_path: str, channels:list[str], grouped: bool) -> bool:
    meta = read_edf_meta(edf_path)
    available_channels = set(meta["signals"])
    requested = get_channels(channels, grouped=grouped, include_quality=False, normalize=True, sample_frequency=SAMPLE_FREQUENCY, override_normalize={"chin_emg":None, "ECG":None, "Pulse Waveform":None}) # TODO this has to be set in two places now
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

def list_patients(source_root: str, channels: list[str], grouped: bool, dry_run: bool) -> list[str]:
    patients = [
        patient for patient in get_edf_files_in_repo(source_root, recursive=True) if not is_pap_patient(patient) and has_required_channels(patient, channels, grouped)
    ]
    return patients[:2] if dry_run else patients

def build_model_and_trainer(train_dataset, model, scaler, grouped,arousal_weight, dry):
    n_channels = len(train_dataset.get_input_channels())
    if model == "utime-big":
        model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=n_channels,
            classes=TARGET_CLASSES, #["arousal"],
            sampling_frequency=SAMPLE_FREQUENCY,
            channel = [64, 128, 128, 256],
            kernel = [5, 5, 3, 3],
            maxpool = [5, 5, 3, 3],
            norm="channel",
            activation="elu",
            mlp_size=512,
            dropout_p=0,
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(7,n_channels)])] if not grouped else [RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(1, n_channels)])] if scaler else None
        )
    elif model == "utime-small":
        model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=n_channels,
            classes=TARGET_CLASSES, #["arousal"],
            sampling_frequency=SAMPLE_FREQUENCY,
            channel = [64, 128, 256],
            kernel = [5, 3, 3],
            maxpool = [5, 3, 3],
            norm="channel",
            activation="elu",
            mlp_size=64,
            dropout_p=0,
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(7,n_channels)])] if not grouped else [RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(1, n_channels)])] if scaler else None 
        )
    elif model == "multi":
        if grouped:
            eeg_channels = ["EEG"]
        else:
            eeg_channels = ["C3-M2", "C4-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"]
        
        aux_channels = [channel for channel in train_dataset.get_input_channels() if channel not in eeg_channels]
        if "EEG" not in train_dataset.get_input_channels() or len(aux_channels) == 0:
            raise ValueError("Multimodel requires at least one EEG and one non-EEG input channel.")

        aux_model = UTime(
            ts_len=train_dataset.get_timeseries_len(),
            n_channels=len(aux_channels),
            classes=None,
            sampling_frequency=SAMPLE_FREQUENCY,
            channel=[32, 64, 128],
            maxpool=[8, 6, 4],
            kernel=[5, 3, 3],
            norm="channel",
            mlp_size=64,
            dropout_p=0,
            activation="elu",
            preprocessors=[RobustScaler(lower_quantile=0.1, upper_quantile=0.9,channels=[i for i in range(1, n_channels)])] if scaler else None 
        )
        eeg_model = SleepTransformer(
            classes=None,
            n_channels=len(eeg_channels),
            epoch_seq_len=5
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
        raise ValueError(f"Did not recoginize model {model}")
    
    trainer = MulticlassTrainer(
        epochs=EPOCHS if not dry else 2,
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
        class_weights={"no_arousal":1, "arousal":arousal_weight}
    )
    return model, trainer


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an arousal model.")
    parser.add_argument("--grouped", action="store_true", help="Randomly sample one available EEG/EOG channel into a grouped EEG input.")
    parser.add_argument("--channels", nargs='+', default=["eeg", "eog", "chin_emg", "ECG"], help="List of channels.")
    parser.add_argument("--scaler", action="store_true", help="If RobustScaler should be used.")
    parser.add_argument("--clean", action="store_true", help="If data should be filtered for artifacts / wake.")
    parser.add_argument("--arousal_weight", type=int, default=1, help="Weight of arousals.")
    parser.add_argument("--model", type=str, default="utime-big", help="What model to use.")
    parser.add_argument("--id", type=str, default="", help="ID of the experiment")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of train patients reserved for validation. Set to 0 to disable.")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    experiment_name = f"arousal_{args.id}" 
    if args.dry:
        logger.info("Performing dry run to test pipeline!")
        experiment_name += "-dev"
    
    os.makedirs(os.path.join("results", "arousal", experiment_name), exist_ok=True)
    logger.set_log_file(os.path.join("results", "arousal", experiment_name, "output.log"))

    train_patients = list_patients(TRAIN_ROOT, args.channels, args.grouped, args.dry)
    test_patients = list_patients(TEST_ROOT, args.channels, args.grouped, args.dry)

    val_dataset = None
    if args.val_frac is not None and args.val_frac > 0:
        train_patients, val_patients = random_split(train_patients, test_frac=args.val_frac)
    else:
        val_patients = []

    # with suppress_stdout_logging(logger):
    logger.context("TRAIN")
    train_dataset = build_dataset(
        train_patients,
        args.channels,
        args.clean,
        args.grouped,
    )
    logger.uncontext()
    if len(val_patients) > 0:
        logger.context("VAL")
        val_dataset = build_dataset(
            val_patients,
            args.channels,
            args.clean,
            args.grouped,
        )
        logger.uncontext()
    logger.context("TEST")
    test_dataset = build_dataset(
        test_patients,
        args.channels,
        args.clean,
        args.grouped,
    )
    logger.uncontext()

    model, trainer = build_model_and_trainer(train_dataset, args.model, args.scaler, args.grouped, args.arousal_weight, args.dry)

    run_result = run(
        RunCfg(
            experiment_name=experiment_name,
            model_name=args.model,
            model=model,
            trainer=trainer,
            train_datasets=[train_dataset],
            val_datasets=[] if val_dataset is None else [val_dataset],
            test_datasets=[("test", test_dataset)],
            batch_size=BATCH_SIZE,
            n_samples=N_SAMPLES,
            num_workers_dataloader=NUM_WORKERS_DATALOADER,
            test_repeats=[1,2] if args.grouped else [1],
            use_energy_tracker=False,
            tags={"model": args.model},
            collate_fn=batch_collate,
            use_mlflow=True,
            log_path=os.path.join("results", "arousal"),
            meta_data=vars(args)
        )
    )

    export_prediction_package(
        os.path.join("results", "arousal", experiment_name, "deploy", experiment_name, ".swmodel"),
        model=run_result.model,
        trainer=run_result.trainer,
        dataset=train_dataset.to_unlabelled(),
        metadata={"experiment_name": experiment_name, **vars(args)},
    )


if __name__ == "__main__":
    main()
