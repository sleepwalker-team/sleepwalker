#!/bin/env python3

from __future__ import annotations

import argparse
import os
from functools import partial

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import torch
import torch.multiprocessing as mp

from sleepwalker.datasets import ChannelConfig, Ruhrlandklinik
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.utils import get_edf_files_in_repo, random_split
from sleepwalker.models.UTime import UTime
from sleepwalker.trainer.MulticlassTrainer import MulticlassTrainer
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.utils.filtering import trim_wake
from sleepwalker.trainer.utils.targets import prepare_multiclass_target
from sleepwalker.utils import logger

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
mp.set_sharing_strategy("file_system")

TRAIN_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/train-test-2023"
TEST_ROOT = "/raid/sleepwalker/ruhrlandklinik/raw/val-2024"
EXPERIMENT_NAME = "lm"
BATCH_SIZE = 64
EPOCHS = 100
N_SAMPLES = 100_000
NUM_WORKERS_DATASET = 8
NUM_WORKERS_DATALOADER = 8
SAMPLE_FREQUENCY = 100
TOTAL_INPUT = "10s"
TARGET_RESOLUTION = "1s"
STRIDE = "1s"
SLEEP_PERCENTAGE = 0.5
PATIENT_FILTER_QUANTILE = 0.05
TARGET_CLASSES = ["lm", "no lm"]
SLEEP_LABELS = ["n1", "n2", "n3", "rem"]

EVENT_MAPPING = {
    "wach": "wake",
    "n1": "n1",
    "n2": "n2",
    "n3": "n3",
    "rem": "rem",
    "lm": "lm",
    # "plms": "lm", # maybe later -> increase total_input to 500s

    # "plm": "plm",
    # "plm-arousal": "plm",
    # "artefakt": "artifact",
    # "bewegung": "movement",
}

prepare_lm_target = partial(
    prepare_multiclass_target,
    target_classes=TARGET_CLASSES,
    filters=[
        {"columns": SLEEP_LABELS, "percentage": SLEEP_PERCENTAGE, "mode": "min"},
        #{"columns": ["artifact", "movement"], "percentage": 0.0, "mode": "max"},
    ],
)


def build_channels():
    return [
        ChannelConfig(
            name="X Axis",
            #normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, lowcut=5.0, highcut=30.0, notch_freq=50.0),
            group=None,
        ),
        ChannelConfig(
            name="Y Axis",
            #normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, lowcut=5.0, highcut=30.0, notch_freq=50.0),
            group=None,
        ),
        ChannelConfig(
            name="Z Axis",
            #normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, lowcut=5.0, highcut=30.0, notch_freq=50.0),
            group=None,
        ),
        # ChannelConfig(
        #     name="Left Leg",
        #     #normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, lowcut=5.0, highcut=30.0, notch_freq=50.0),
        #     group=None,
        # ),
        # ChannelConfig(
        #     name="Right Leg",
        #     #normalizer=SignalFilterNormalizer(fs=SAMPLE_FREQUENCY, lowcut=5.0, highcut=30.0, notch_freq=50.0),
        #     group=None,
        # ),
        #ChannelConfig(name="Activity", normalizer=None, group=None),
    ]

def prepare_patient(data_df, label_df, label_extra_df, patient=None):
    trimmed = trim_wake(data_df, label_df, label_extra_df)
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df

def prepare_sample(data, quality_data=None, target=None, target_extra=None, patient=None, time=None):
    if float(data.isna().mean().mean()) > 0.05:
        return None

    # for col in ["Left Leg", "Right Leg"]:
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

def create_dataset():
    return Ruhrlandklinik(
        channels=build_channels(),
        sample_frequency=SAMPLE_FREQUENCY,
        event_mapping=EVENT_MAPPING,
        stride=STRIDE,
        prepare_patient=prepare_patient,
        prepare_target=prepare_lm_target,
        prepare_sample=prepare_sample,
        total_input=TOTAL_INPUT,
        target_resolution=TARGET_RESOLUTION,
    )


def summarize_patient_lm_duration(patient, data_df, label_df, label_extra_df):
    if label_df is None or len(label_df) == 0:
        return None

    lm_df = label_df[label_df["Label"] == "lm"]
    if len(lm_df) == 0:
        total_lm_seconds = 0.0
    else:
        total_lm_seconds = float((lm_df["Endtime"] - lm_df["Starttime"]).dt.total_seconds().sum())

    return {
        "patient": patient,
        "total_lm_seconds": total_lm_seconds,
    }


def apply_patient_filters(patients: list[str]) -> list[str]:
    dataset = create_dataset()
    stats_df = dataset.get_patient_stats(patients, summarize_patient_lm_duration, num_workers=NUM_WORKERS_DATASET)
    if len(stats_df) == 0:
        logger.warning("Could not compute LM duration stats for any patients; skipping LM patient filtering.")
        return patients

    lower = float(stats_df["total_lm_seconds"].quantile(PATIENT_FILTER_QUANTILE))
    filtered = stats_df[stats_df["total_lm_seconds"] >= lower]
    kept = filtered["patient"].tolist()
    logger.info(
        f"Filtered LM patients: kept {len(kept)}/{len(patients)} with total_lm_seconds >= {lower:.2f} "
        f"(q={PATIENT_FILTER_QUANTILE:.2f})."
    )
    return kept


def build_dataset(patients: list[str]):
    dataset = create_dataset()
    dataset.classes = list(TARGET_CLASSES)
    dataset.initialize(patients, NUM_WORKERS_DATASET)
    return dataset

def build_model_and_trainer(train_dataset):
    model = UTime(
        ts_len=train_dataset.get_timeseries_len(),
        n_channels=len(train_dataset.get_input_channels()),
        classes=TARGET_CLASSES,
        sampling_frequency=SAMPLE_FREQUENCY,
        channel=[64, 128],
        maxpool=[6, 4],
        kernel=[3, 3],
        norm="channel",
        mlp_size=64,
    )
    trainer = MulticlassTrainer(
        epochs=EPOCHS,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-2, total_iters=50
        ),
        classes=TARGET_CLASSES,
        loss_function=torch.nn.functional.cross_entropy,
        save_every=10,
        # loss_mode="inverse",
        balance_batches=True,
        early_stopping=5
    )
    return model, trainer


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a LM model.")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of train patients reserved for validation. Set to 0 to disable.")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    train_patients = get_edf_files_in_repo(TRAIN_ROOT, recursive=True)
    test_patients = get_edf_files_in_repo(TEST_ROOT, recursive=True)
    if args.dry:
        train_patients = train_patients[:2]
        test_patients = test_patients[:2]

    train_patients = apply_patient_filters(train_patients)
    test_patients = apply_patient_filters(test_patients)

    val_dataset = None
    if args.val_frac is not None and args.val_frac > 0:
        train_patients, val_patients = random_split(train_patients, test_frac=args.val_frac)
    else:
        val_patients = []

    # with suppress_stdout_logging(logger):
    logger.context("TRAIN")
    train_dataset = build_dataset(train_patients)
    logger.uncontext()
    if len(val_patients) > 0:
        logger.context("VAL")
        val_dataset = build_dataset(val_patients)
        logger.uncontext()
    logger.context("TEST")
    test_dataset = build_dataset(test_patients)
    logger.uncontext()

    model, trainer = build_model_and_trainer(train_dataset)

    experiment_name = f"{EXPERIMENT_NAME}" 
    if args.dry:
        logger.info("Performing dry run to test pipeline!")
        experiment_name += "-dev"

    run(
        RunCfg(
            experiment_name=experiment_name,
            model_name="UTime",
            model=model,
            trainer=trainer,
            train_datasets=[train_dataset],
            val_datasets=[] if val_dataset is None else [val_dataset],
            test_datasets=[("test", test_dataset)],
            batch_size=BATCH_SIZE,
            n_samples=N_SAMPLES,
            num_workers_dataloader=NUM_WORKERS_DATALOADER,
            test_repeats=[1],
            use_energy_tracker=False,
            tags={"model": "UTime"},
            collate_fn=batch_collate,
        )
    )


if __name__ == "__main__":
    main()
