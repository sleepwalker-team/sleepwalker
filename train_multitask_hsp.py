#!/usr/bin/env python3
"""Shared-backbone HSP multitask baseline for the expert-interface paper."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from sleepwalker.core.signal import read_edf_meta
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.HSP import HSP, get_channels, get_hsp_annotation_path
from sleepwalker.deployment import load_expert_package
from sleepwalker.models.MetaModel import MetaModel, MetaModelEntry
from sleepwalker.models.SleepTransformer import SleepTransformer
from sleepwalker.models.UTime import UTime
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.trainer.Run import RunCfg, run, seed_everything
from sleepwalker.trainer.utils.filtering import trim_event
from sleepwalker.trainer.utils.metrics import (
    cohen_kappa_from_confusion_matrix,
    f1_score_from_confusion_matrix,
)
from sleepwalker.trainer.utils.splits import load_split


TASK_CONFIG = {
    "sleep": {
        "labels": ["wake", "n1", "n2", "n3", "rem"],
        "default": None,
        "target_resolution": "30s",
        "loss_function": torch.nn.functional.cross_entropy,
        "loss_mode": "inverse-log",
    },
    "arousal": {
        "labels": ["arousal", "no arousal"],
        "default": "no arousal",
        "target_resolution": "1s",
        "loss_function": torch.nn.functional.cross_entropy,
        "loss_mode": "inverse-log",
    },
    "breathing": {
        "labels": ["apnea", "hypopnea", "regular breathing"],
        "default": "regular breathing",
        "target_resolution": "5s",
        "loss_function": torch.nn.functional.cross_entropy,
        "loss_mode": "inverse-log",
    },
    "desaturation": {
        "labels": ["desaturation", "no desaturation"],
        "default": "no desaturation",
        "target_resolution": "10s",
        "loss_function": torch.nn.functional.cross_entropy,
        "loss_mode": "inverse-log",
    },
}
NORMALIZED_TASK_CONFIG = MultiLabelTrainer.normalize_task_config(TASK_CONFIG)
EVENT_MAPPING = {
    "wake": "wake",
    "n1": "n1",
    "n2": "n2",
    "n3": "n3",
    "rem": "rem",
    "arousal": "arousal",
    "apnea": "apnea",
    "obstructive-apnea": "apnea",
    "mixed-apnea": "apnea",
    "central-apnea": "apnea",
    "hypopnea": "hypopnea",
    "desaturation": "desaturation",
}
CHANNEL_GROUPS = ["eeg", "eog", "chin_emg", "abdomen", "chest", "airflow", "spo2"]


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected top-level mapping in {path}.")
    allowed = {
        "run_id", "seed", "split", "output_dir", "sample_frequency",
        "total_input", "stride", "limits", "training", "model",
        "assume_units_if_missing",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unknown monolithic config keys in {path}: {unknown}")
    required = {
        "run_id", "seed", "split", "output_dir", "sample_frequency",
        "total_input", "training", "model",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing monolithic config keys in {path}: {missing}")
    return config


def prepare_patient(data_df, label_df, label_extra_df, patient=None):
    trimmed = trim_event(
        data_df,
        label_df,
        label_extra_df,
        keep_events=["wake", "n1", "n2", "n3", "rem"],
    )
    if trimmed is None:
        return None
    label_df, label_extra_df = trimmed
    return data_df, label_df, label_extra_df


def prepare_sample(data, target=None, target_extra=None, patient=None, time=None, **_kwargs):
    if data.isna().any().any():
        return None
    return {
        "data": torch.from_numpy(data.values).float(),
        "target": target,
        "patient": patient,
        "time": time,
    }


def build_dataset_template(config: dict[str, Any]) -> HSP:
    sample_frequency = float(config["sample_frequency"])
    return HSP(
        channels=get_channels(
            CHANNEL_GROUPS,
            grouped=True,
            normalize=True,
            sample_frequency=sample_frequency,
            override_normalize={"chin_emg": None, "leg_emg": None},
        ),
        sample_frequency=sample_frequency,
        event_mapping=EVENT_MAPPING,
        remove_unmapped_events=True,
        prepare_patient=prepare_patient,
        prepare_target=partial(
            MultiLabelTrainer.prepare_target,
            task_config=NORMALIZED_TASK_CONFIG,
        ),
        prepare_sample=prepare_sample,
        total_input=str(config["total_input"]),
        target_resolution="30s",
        stride=str(config.get("stride", "30s")),
        assume_units_if_missing=bool(config.get("assume_units_if_missing", False)),
    )


def has_required_channels(edf_path: str, config: dict[str, Any]) -> bool:
    available = set(read_edf_meta(edf_path)["signals"])
    requested = get_channels(
        CHANNEL_GROUPS,
        grouped=True,
        normalize=True,
        sample_frequency=float(config["sample_frequency"]),
        override_normalize={"chin_emg": None, "leg_emg": None},
    )
    alternatives: dict[str, list[str]] = {}
    for channel in requested:
        alternatives.setdefault(channel.group or channel.name, []).append(channel.name)
    return all(any(name in available for name in names) for names in alternatives.values())


def has_annotation(edf_path: str) -> bool:
    return get_hsp_annotation_path(edf_path) is not None


def select_patients(records: list[str], config: dict[str, Any]) -> list[str]:
    return [
        record
        for record in records
        if has_required_channels(record, config) and has_annotation(record)
    ]


def build_model(config: dict[str, Any], input_channels: list[str]) -> MetaModel:
    sample_frequency = float(config["sample_frequency"])
    total_samples = int(pd.to_timedelta(config["total_input"]).total_seconds() * sample_frequency)
    eeg_channels = ["EEG"]
    auxiliary_channels = [channel for channel in input_channels if channel not in eeg_channels]
    if not auxiliary_channels or any(channel not in input_channels for channel in eeg_channels):
        raise ValueError(
            "The monolithic baseline requires grouped EEG and at least one non-EEG channel."
        )

    model_cfg = config["model"]
    auxiliary = UTime(
        ts_len=total_samples,
        n_channels=len(auxiliary_channels),
        sampling_frequency=pd.to_timedelta(1.0 / sample_frequency, unit="s"),
        classes=None,
        channel=list(model_cfg["auxiliary_channels"]),
        maxpool=list(model_cfg["auxiliary_maxpool"]),
        kernel=list(model_cfg["auxiliary_kernel"]),
        norm=str(model_cfg["auxiliary_norm"]),
        mlp_size=int(model_cfg["auxiliary_feature_dim"]),
        activation="elu",
        dropout_p=0,
    )
    stft_frames = total_samples // 64 + 1
    available_epochs = stft_frames // 29
    epoch_seq_len = min(21, available_epochs)
    if epoch_seq_len % 2 == 0:
        epoch_seq_len -= 1
    if epoch_seq_len < 1:
        raise ValueError(
            f"total_input={config['total_input']} is too short for SleepTransformer."
        )
    eeg = SleepTransformer(
        ts_len=total_samples,
        classes=None,
        n_channels=len(eeg_channels),
        epoch_seq_len=epoch_seq_len,
    )
    return MetaModel(
        task_config=NORMALIZED_TASK_CONFIG,
        input_channels=input_channels,
        models=[
            MetaModelEntry(eeg, eeg_channels),
            MetaModelEntry(auxiliary, auxiliary_channels),
        ],
    )


def build_trainer(config: dict[str, Any]) -> MultiLabelTrainer:
    return MultiLabelTrainer(
        epochs=int(config["training"]["epochs"]),
        optimizer=lambda model: torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        ),
        task_config=NORMALIZED_TASK_CONFIG,
        condition_task="sleep",
        condition_labels=["n1", "n2", "n3", "rem"],
        conditioned_tasks=["arousal", "breathing", "desaturation"],
        device=str(config["training"]["device"]),
        warmup_device=str(config["training"]["warmup_device"]),
        save_every=1,
        early_stopping=int(config["training"]["patience"]),
        log_batches=False,
    )


def build_expert_components(config: dict[str, Any]) -> dict[str, Any]:
    dataset = build_dataset_template(config)
    return {
        "dataset_template": dataset,
        "model": build_model(config, dataset.get_input_channels()),
        "trainer": build_trainer(config),
    }


class EvenSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_items: int | None):
        self.dataset = dataset
        if max_items is None or len(dataset) <= int(max_items):
            self.indices = np.arange(len(dataset), dtype=np.int64)
        else:
            self.indices = np.linspace(0, len(dataset) - 1, int(max_items), dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def paths_for_split(split: dict[str, Any], name: str, limit: int | None) -> list[str]:
    task_paths = [
        set(split["tasks"][task]["edf_files"][name])
        for task in ["sleep", "arousal", "breathing", "desaturation"]
    ]
    common = sorted(set.intersection(*task_paths))
    if limit is not None:
        common = common[: int(limit)]
    if not common:
        raise ValueError(f"No four-task HSP records in split '{name}'.")
    return common


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def collect_record(config: dict[str, Any], record: dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    experiment_name = f"paper_monolithic_hsp_{config['run_id']}"
    final_dir = output_dir / experiment_name / "final"
    packages = [
        path for path in final_dir.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    ] if final_dir.exists() else []
    if not packages:
        raise ValueError(f"No exported monolithic package in {final_dir}.")
    package_path = max(packages, key=lambda path: path.stat().st_mtime_ns)
    load_expert_package(package_path, map_location="cpu")

    summary = {
        "experiment_name": experiment_name,
        "package_path": str(package_path),
        "package_reload_ok": True,
        "test_records": [record],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "monolithic_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2),
        encoding="utf-8",
    )

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Task & Accuracy & Macro-F1 & $\kappa$ & Samples \\",
        r"\midrule",
    ]
    for task, values in record["test_cm"].items():
        matrix = np.asarray(values, dtype=np.int64)
        total = int(matrix.sum())
        accuracy = float(np.trace(matrix) / total) if total else 0.0
        lines.append(
            f"{task.replace('_', r'\_')} & {accuracy:.3f} & "
            f"{f1_score_from_confusion_matrix(matrix, macro=True):.3f} & "
            f"{cohen_kappa_from_confusion_matrix(matrix):.3f} & {total} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    artifact_tag = str(config["run_id"]).replace("-", "_")
    table_path = Path("paper") / "tables" / f"monolithic_results_{artifact_tag}.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines), encoding="utf-8")


def command_collect(config: dict[str, Any]) -> None:
    results_path = Path(config["output_dir"]) / "results.jsonl"
    if not results_path.exists():
        raise ValueError(f"Missing monolithic results {results_path}.")
    experiment_name = f"paper_monolithic_hsp_{config['run_id']}"
    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matching = [record for record in records if record.get("name") == experiment_name]
    if not matching:
        raise ValueError(f"No record for '{experiment_name}' in {results_path}.")
    collect_record(config, matching[-1])


def command_validate(config: dict[str, Any]) -> None:
    """Run one synthetic forward/backward pass before an expensive full run."""
    seed_everything(int(config["seed"]))
    components = build_expert_components(config)
    model = components["model"]
    device = torch.device(str(config["training"]["device"]))
    model.to(device)
    shape, _metadata = model.input_spec()
    batch_size = int(config["training"]["batch_size"])
    x = torch.zeros((batch_size, *shape[1:]), dtype=torch.float32, device=device)
    output = model(x)
    expected_tasks = set(TASK_CONFIG)
    if set(output) != expected_tasks:
        raise ValueError(
            f"Shared model emitted tasks {sorted(output)}, expected {sorted(expected_tasks)}."
        )
    loss = torch.zeros((), device=device)
    for task, logits in output.items():
        expected = (
            batch_size,
            int(NORMALIZED_TASK_CONFIG[task]["n_steps"]),
            len(NORMALIZED_TASK_CONFIG[task]["labels"]),
        )
        if tuple(logits.shape) != expected:
            raise ValueError(
                f"Task '{task}' emitted {tuple(logits.shape)}, expected {expected}."
            )
        loss = loss + logits.square().mean()
    loss.backward()
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "device": str(device),
                "input_shape": list(x.shape),
                "output_shapes": {
                    task: list(logits.shape) for task, logits in output.items()
                },
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "peak_memory_bytes": peak_bytes,
            },
            indent=2,
        )
    )


def command_train(config: dict[str, Any]) -> None:
    seed_everything(int(config["seed"]))
    split = load_split(config["split"])
    limits = config.get("limits", {})
    datasets = {}
    for split_name in ["train", "val", "test"]:
        selected = select_patients(
            paths_for_split(split, split_name, limits.get(f"{split_name}_patients")),
            config,
        )
        dataset = build_dataset_template(config)
        dataset.initialize(
            selected,
            int(config["training"]["num_workers_dataset"]),
        )
        datasets[split_name] = dataset
    datasets["test"] = EvenSubset(datasets["test"], limits.get("test_windows"))

    components = build_expert_components(config)
    experiment_name = f"paper_monolithic_hsp_{config['run_id']}"
    result = run(
        RunCfg(
            experiment_name=experiment_name,
            model_name="sleeptransformer-utime",
            model=components["model"],
            trainer=components["trainer"],
            train_datasets=[datasets["train"]],
            val_datasets=[datasets["val"]],
            test_datasets=[("HSP", datasets["test"])],
            batch_size=int(config["training"]["batch_size"]),
            n_samples=int(config["training"]["n_samples"]),
            num_workers_dataloader=int(config["training"]["num_workers_dataloader"]),
            n_samples_test=limits.get("test_windows"),
            test_repeats=[1],
            use_mlflow=False,
            log_path=str(Path(config["output_dir"])),
            tags={"task": "multitask", "dataset": "hsp", "baseline": "monolithic"},
            collate_fn=batch_collate,
            meta_data=config,
            expert_task="multitask",
            inference_dataset=components["dataset_template"],
        )
    )
    collect_record(config, result.test_results[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.collect_only and args.validate_only:
        parser.error("--collect-only and --validate-only are mutually exclusive.")
    config = read_config(args.config)
    if args.collect_only:
        command_collect(config)
    elif args.validate_only:
        command_validate(config)
    else:
        command_train(config)


if __name__ == "__main__":
    main()
