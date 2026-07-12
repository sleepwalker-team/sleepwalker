#!/usr/bin/env python3
"""End-to-end smoke pipeline for the structured-expert paper.

The goal is not model quality. The goal is to keep the paper pipeline honest:
stable HSP splits, tiny expert training, package reload checks, evaluation
records, and paper-ready tables/plots should all exist before long runs start.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.datasets.HSP import (
    get_annotated_hsp_edf_files,
    get_hsp_annotation_label_counts,
    get_hsp_annotation_path,
)
from sleepwalker.core.signal import read_edf_meta
from sleepwalker.deployment import load_expert_package
from sleepwalker.trainer.Run import RunCfg, run
from sleepwalker.trainer.utils.metrics import (
    cohen_kappa_from_confusion_matrix,
    f1_score_from_confusion_matrix,
)
from sleepwalker.utils import logger

import train_arousal_hsp
import train_event_hsp
import train_sleep
from hsp_filter_report import TASK_DEFAULTS, apply_task_defaults, requested_channel_groups


PAPER_TASKS = ["sleep", "arousal", "breathing", "desaturation"]
EVENT_TASKS = {"breathing", "desaturation"}


def read_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a top-level mapping in {path}")
    return cfg


def subject_id(edf_path: str) -> str:
    match = re.search(r"/(sub-[^/]+)/", edf_path)
    if match:
        return match.group(1)
    return Path(edf_path).stem


def split_name_for_subject(subject: str, seed: str, fractions: dict[str, float]) -> str:
    value = int(hashlib.sha1(f"{seed}:{subject}".encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
    train_cutoff = float(fractions["train"])
    val_cutoff = train_cutoff + float(fractions["val"])
    if value < train_cutoff:
        return "train"
    if value < val_cutoff:
        return "val"
    return "test"


def mapped_positive_labels_for_task(task: str) -> list[str]:
    return list(TASK_DEFAULTS[task]["positive_labels"])


def has_mapped_task_label(label_counts: dict[str, int], task: str) -> bool:
    return any(label_counts.get(label, 0) > 0 for label in mapped_positive_labels_for_task(task))


def task_candidate_cfg(root: str, task: str) -> dict[str, Any]:
    cfg = {
        "root": root,
        "task": task,
        "sample_frequency": 100,
        "annotated_only": True,
    }
    return apply_task_defaults(cfg)


def task_requirements(root: str) -> dict[str, dict[str, Any]]:
    requirements = {}
    for task in PAPER_TASKS:
        cfg = task_candidate_cfg(root, task)
        requirements[task] = {
            "required_channel_groups": requested_channel_groups(cfg),
            "required_any_labels": set(cfg.get("required_any_labels") or []),
            "min_duration_s": cfg.get("min_duration_s"),
        }
    return requirements


def usable_tasks_for_edf(args: tuple[str, dict[str, dict[str, Any]]]) -> tuple[str, list[str]]:
    edf_path, requirements = args
    annotation_path = get_hsp_annotation_path(edf_path)
    if annotation_path is None:
        return edf_path, []
    try:
        label_counts = get_hsp_annotation_label_counts(annotation_path)
        labels = set(label_counts)
        meta = read_edf_meta(edf_path)
        available_channels = set(meta["signals"])
    except Exception:
        return edf_path, []

    usable = []
    for task, task_req in requirements.items():
        min_duration_s = task_req.get("min_duration_s")
        if min_duration_s is not None and float(meta["duration_s"]) < float(min_duration_s):
            continue
        required_any_labels = task_req["required_any_labels"]
        if required_any_labels and labels.isdisjoint(required_any_labels):
            continue
        missing_channel_group = any(
            not any(name in available_channels for name in names)
            for names in task_req["required_channel_groups"].values()
        )
        if missing_channel_group:
            continue
        if task in {"arousal", "breathing", "desaturation"} and not has_mapped_task_label(label_counts, task):
            continue
        usable.append(task)
    return edf_path, usable


def build_hsp_split(cfg: dict[str, Any]) -> dict[str, Any]:
    root = str(cfg["root"])
    seed = str(cfg.get("seed", "sleepwalker-paper-v0"))
    fractions = dict(cfg.get("split", {"train": 0.8, "val": 0.1, "test": 0.1}))
    split: dict[str, Any] = {
        "root": root,
        "seed": seed,
        "method": "subject-hash",
        "fractions": fractions,
        "tasks": {},
    }
    paths_by_task = {task: [] for task in PAPER_TASKS}
    edf_files = get_annotated_hsp_edf_files(root, recursive=True)
    requirements = task_requirements(root)
    workers = int(cfg.get("num_workers_split", cfg.get("training", {}).get("num_workers_dataset", 2)))
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            iterator = pool.imap_unordered(usable_tasks_for_edf, [(path, requirements) for path in edf_files])
            for edf_path, usable_tasks in iterator:
                for task in usable_tasks:
                    paths_by_task[task].append(edf_path)
    else:
        for edf_path in edf_files:
            _, usable_tasks = usable_tasks_for_edf((edf_path, requirements))
            for task in usable_tasks:
                paths_by_task[task].append(edf_path)

    for task in PAPER_TASKS:
        task_splits = {"train": [], "val": [], "test": []}
        for edf_path in sorted(paths_by_task[task]):
            split_name = split_name_for_subject(subject_id(edf_path), seed, fractions)
            task_splits[split_name].append(edf_path)
        split["tasks"][task] = {
            "edf_files": task_splits,
            "counts": {key: len(value) for key, value in task_splits.items()},
            "total_edf_files": len(paths_by_task[task]),
        }
    return split


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def limited(paths: list[str], limit: int | None) -> list[str]:
    if limit is None:
        return list(paths)
    return list(paths[: int(limit)])


def split_paths(split: dict[str, Any], task: str, cfg: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    limits = cfg.get("limits", {})
    task_split = split["tasks"][task]["edf_files"]
    return (
        limited(task_split["train"], limits.get("train_edf_files")),
        limited(task_split["val"], limits.get("val_edf_files")),
        limited(task_split["test"], limits.get("test_edf_files")),
    )


class LimitedDataset:
    def __init__(self, dataset, max_items: int | None):
        self.dataset = dataset
        self.max_items = None if max_items is None else int(max_items)

    def __len__(self):
        if self.max_items is None:
            return len(self.dataset)
        return min(len(self.dataset), self.max_items)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def limited_test_dataset(dataset, cfg: dict[str, Any]):
    max_windows = cfg.get("limits", {}).get("test_windows")
    if dataset is None or max_windows is None:
        return dataset
    return LimitedDataset(dataset, int(max_windows))


def latest_final_package(log_path: Path, experiment_name: str) -> str | None:
    final_dir = log_path / experiment_name / "final"
    if not final_dir.exists():
        return None
    candidates = sorted(path for path in final_dir.iterdir() if path.is_dir())
    return str(candidates[-1]) if candidates else None


def train_sleep_expert(split: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    train_paths, val_paths, test_paths = split_paths(split, "sleep", cfg)
    training = cfg["training"]
    train_sleep.NUM_WORKERS_DATASET = int(training["num_workers_dataset"])
    train_sleep.NUM_WORKERS_DATALOADER = int(training["num_workers_dataloader"])

    model_name = "sleeptransformer"
    total_input = train_sleep.MODEL_CFG[model_name]["total_input"]
    train_dataset = train_sleep.build_dataset("hsp", train_paths, grouped=True, total_input=total_input)
    val_dataset = train_sleep.build_dataset("hsp", val_paths, grouped=True, total_input=total_input) if val_paths else None
    test_dataset = train_sleep.build_dataset("hsp", test_paths, grouped=True, total_input=total_input) if test_paths else None
    test_dataset = limited_test_dataset(test_dataset, cfg)
    model, trainer = train_sleep.build_model_and_trainer(train_dataset, model_name, dry_run=False)
    trainer.epochs = int(training["epochs"])
    trainer.save_every = 1

    experiment_name = "paper_sleep_hsp_smoke"
    log_path = Path(cfg["output_dir"]) / "sleep"
    builder_config = {
        "model": model_name,
        "dataset": "hsp",
        "grouped": True,
        "epochs": trainer.epochs,
        "total_input": total_input,
    }
    result = run(
        RunCfg(
            experiment_name=experiment_name,
            model_name=model_name,
            model=model,
            trainer=trainer,
            train_datasets=[train_dataset],
            val_datasets=[] if val_dataset is None else [val_dataset],
            test_datasets=[] if test_dataset is None else [("HSP", test_dataset)],
            batch_size=int(training["batch_size"]),
            n_samples=int(training["n_samples"]),
            num_workers_dataloader=int(training["num_workers_dataloader"]),
            test_repeats=[1],
            use_energy_tracker=False,
            use_mlflow=bool(training.get("use_mlflow", False)),
            log_path=str(log_path),
            tags={"task": "sleep", "dataset": "hsp"},
            collate_fn=batch_collate,
            meta_data={"pipeline": "paper-smoke", "task": "sleep", **builder_config},
            expert_name=experiment_name,
            expert_task="sleep",
            expert_builder={"module": "train_sleep", "function": "build_expert_components", "config": builder_config},
            expert_dataset_template=train_sleep.build_expert_components(builder_config)["dataset_template"],
        )
    )
    return run_summary("sleep", experiment_name, log_path, result)


def train_arousal_expert(split: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    train_paths, val_paths, test_paths = split_paths(split, "arousal", cfg)
    training = cfg["training"]
    task_cfg = dict(train_arousal_hsp.DEFAULT_CONFIG)
    task_cfg.update(
        {
            "root": cfg["root"],
            "epochs": int(training["epochs"]),
            "n_samples": int(training["n_samples"]),
            "batch_size": int(training["batch_size"]),
            "num_workers_dataset": int(training["num_workers_dataset"]),
            "num_workers_dataloader": int(training["num_workers_dataloader"]),
            "use_mlflow": bool(training.get("use_mlflow", False)),
            "dry": False,
            "grouped": True,
            "scaler": True,
            "arousal_weight": 10,
        }
    )
    train_dataset = train_arousal_hsp.build_dataset(train_paths, task_cfg)
    val_dataset = train_arousal_hsp.build_dataset(val_paths, task_cfg) if val_paths else None
    test_dataset = train_arousal_hsp.build_dataset(test_paths, task_cfg) if test_paths else None
    test_dataset = limited_test_dataset(test_dataset, cfg)
    model, trainer = train_arousal_hsp.build_model_and_trainer(train_dataset, task_cfg)
    trainer.epochs = int(training["epochs"])
    trainer.save_every = 1

    experiment_name = "paper_arousal_hsp_smoke"
    log_path = Path(cfg["output_dir"]) / "arousal"
    builder_config = {
        "sample_frequency": task_cfg["sample_frequency"],
        "target_resolution": task_cfg["target_resolution"],
        "stride": task_cfg["stride"],
        "channels": list(task_cfg["channels"]),
        "clean": bool(task_cfg.get("clean", False)),
        "grouped": task_cfg["grouped"],
        "total_input": task_cfg["total_input"],
        "model": task_cfg["model"],
        "scaler": task_cfg["scaler"],
        "arousal_weight": task_cfg["arousal_weight"],
        "balance_batches": task_cfg.get("balance_batches", False),
        "balance_gamma": task_cfg.get("balance_gamma", 0.75),
        "epochs": trainer.epochs,
        "num_workers_dataset": task_cfg["num_workers_dataset"],
        "annotated_only": task_cfg.get("annotated_only", True),
    }
    result = run(
        RunCfg(
            experiment_name=experiment_name,
            model_name=task_cfg["model"],
            model=model,
            trainer=trainer,
            train_datasets=[train_dataset],
            val_datasets=[] if val_dataset is None else [val_dataset],
            test_datasets=[] if test_dataset is None else [("HSP", test_dataset)],
            batch_size=int(training["batch_size"]),
            n_samples=int(training["n_samples"]),
            num_workers_dataloader=int(training["num_workers_dataloader"]),
            test_repeats=[1],
            use_energy_tracker=False,
            use_mlflow=bool(training.get("use_mlflow", False)),
            log_path=str(log_path),
            tags={"task": "arousal", "dataset": "hsp"},
            collate_fn=batch_collate,
            meta_data={"pipeline": "paper-smoke", "task": "arousal", **builder_config},
            expert_name=experiment_name,
            expert_task="arousal",
            expert_builder={
                "module": "train_arousal_hsp",
                "function": "build_expert_components",
                "config": builder_config,
            },
            expert_dataset_template=train_arousal_hsp.build_expert_components(builder_config)["dataset_template"],
        )
    )
    return run_summary("arousal", experiment_name, log_path, result)


def train_event_expert(task: str, split: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    train_paths, val_paths, test_paths = split_paths(split, task, cfg)
    training = cfg["training"]
    task_cfg = dict(train_event_hsp.DEFAULT_CONFIG)
    task_cfg.update(
        {
            "root": cfg["root"],
            "task": task,
            "epochs": int(training["epochs"]),
            "n_samples": int(training["n_samples"]),
            "batch_size": int(training["batch_size"]),
            "num_workers_dataset": int(training["num_workers_dataset"]),
            "num_workers_dataloader": int(training["num_workers_dataloader"]),
            "use_mlflow": bool(training.get("use_mlflow", False)),
            "dry": False,
        }
    )
    defaults = train_event_hsp.TASKS[task]
    for key in ["channels", "target_resolution", "stride", "total_input"]:
        task_cfg.setdefault(key, defaults[key])
    task_cfg.setdefault("class_weights", defaults["class_weights"])

    train_dataset = train_event_hsp.build_dataset(train_paths, task_cfg)
    val_dataset = train_event_hsp.build_dataset(val_paths, task_cfg) if val_paths else None
    test_dataset = train_event_hsp.build_dataset(test_paths, task_cfg) if test_paths else None
    test_dataset = limited_test_dataset(test_dataset, cfg)
    model, trainer = train_event_hsp.build_model_and_trainer(train_dataset, task_cfg)
    trainer.epochs = int(training["epochs"])
    trainer.save_every = 1

    experiment_name = f"paper_{task}_hsp_smoke"
    log_path = Path(cfg["output_dir"]) / task
    builder_config = {
        "task": task_cfg["task"],
        "channels": list(task_cfg["channels"]),
        "sample_frequency": task_cfg["sample_frequency"],
        "stride": task_cfg["stride"],
        "total_input": task_cfg["total_input"],
        "target_resolution": task_cfg["target_resolution"],
        "sleep_percentage": task_cfg["sleep_percentage"],
        "grouped": task_cfg["grouped"],
        "model": task_cfg["model"],
        "scaler": task_cfg["scaler"],
        "class_weights": task_cfg.get("class_weights", defaults["class_weights"]),
        "epochs": trainer.epochs,
    }
    result = run(
        RunCfg(
            experiment_name=experiment_name,
            model_name=task_cfg["model"],
            model=model,
            trainer=trainer,
            train_datasets=[train_dataset],
            val_datasets=[] if val_dataset is None else [val_dataset],
            test_datasets=[] if test_dataset is None else [("HSP", test_dataset)],
            batch_size=int(training["batch_size"]),
            n_samples=int(training["n_samples"]),
            num_workers_dataloader=int(training["num_workers_dataloader"]),
            test_repeats=[1],
            use_energy_tracker=False,
            use_mlflow=bool(training.get("use_mlflow", False)),
            log_path=str(log_path),
            tags={"task": task, "dataset": "hsp"},
            collate_fn=batch_collate,
            meta_data={"pipeline": "paper-smoke", "task": task, **builder_config},
            expert_name=experiment_name,
            expert_task=task,
            expert_builder={"module": "train_event_hsp", "function": "build_expert_components", "config": builder_config},
            expert_dataset_template=train_event_hsp.build_expert_components(builder_config)["dataset_template"],
        )
    )
    return run_summary(task, experiment_name, log_path, result)


def run_summary(task: str, experiment_name: str, log_path: Path, result) -> dict[str, Any]:
    package_path = latest_final_package(log_path, experiment_name)
    package_ok = False
    package_model = None
    if package_path is not None:
        loaded = load_expert_package(package_path, map_location="cpu")
        package_model = type(loaded.model).__name__
        package_ok = True
    return {
        "task": task,
        "experiment_name": experiment_name,
        "log_path": str(log_path / experiment_name),
        "package_path": package_path,
        "package_reload_ok": package_ok,
        "package_model": package_model,
        "test_records": result.test_records,
    }


def _summary_can_resume(summary: dict[str, Any]) -> bool:
    package_path = summary.get("package_path")
    if not package_path:
        return False
    try:
        load_expert_package(package_path, map_location="cpu")
    except Exception:
        return False
    return bool(summary.get("package_reload_ok"))


def train_smoke(split: dict[str, Any], cfg: dict[str, Any], output_dir: Path | None = None) -> list[dict[str, Any]]:
    runners = [
        ("sleep", lambda: train_sleep_expert(split, cfg)),
        ("arousal", lambda: train_arousal_expert(split, cfg)),
        ("breathing", lambda: train_event_expert("breathing", split, cfg)),
        ("desaturation", lambda: train_event_expert("desaturation", split, cfg)),
    ]
    partial_path = None if output_dir is None else output_dir / "run_summaries.partial.json"
    existing: dict[str, dict[str, Any]] = {}
    if partial_path is not None and partial_path.exists() and bool(cfg.get("resume", True)):
        for summary in json.loads(partial_path.read_text(encoding="utf-8")):
            if _summary_can_resume(summary):
                existing[summary["task"]] = summary

    summaries = []
    for task, runner in runners:
        summary = existing.get(task)
        if summary is None:
            summary = runner()
        summaries.append(summary)
        if partial_path is not None:
            write_json(partial_path, summaries)
    return summaries


def metric_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        records = summary.get("test_records") or []
        if not records:
            rows.append(
                {
                    "task": summary["task"],
                    "model": summary.get("package_model") or "",
                    "dataset": "HSP",
                    "test_loss": "",
                    "accuracy": "",
                    "macro_f1": "",
                    "kappa": "",
                    "package_reload_ok": summary["package_reload_ok"],
                    "package_path": summary.get("package_path") or "",
                }
            )
            continue
        for record in records:
            cm = np.asarray(record["test_cm"])
            total = cm.sum()
            accuracy = float(cm.trace() / total) if total > 0 else 0.0
            rows.append(
                {
                    "task": summary["task"],
                    "model": record.get("model", summary.get("package_model") or ""),
                    "dataset": record.get("dataset", "HSP"),
                    "test_loss": float(record["test_loss"]),
                    "accuracy": accuracy,
                    "macro_f1": float(f1_score_from_confusion_matrix(cm, macro=True)) if total > 0 else 0.0,
                    "kappa": float(cohen_kappa_from_confusion_matrix(cm)) if total > 0 else 0.0,
                    "package_reload_ok": summary["package_reload_ok"],
                    "package_path": summary.get("package_path") or "",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Task & Model & Dataset & Loss & Acc. & Macro-F1 & $\\kappa$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        def fmt(value: Any) -> str:
            if value == "":
                return "--"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)

        lines.append(
            f"{row['task']} & {row['model']} & {row['dataset']} & "
            f"{fmt(row['test_loss'])} & {fmt(row['accuracy'])} & {fmt(row['macro_f1'])} & {fmt(row['kappa'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_metric_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty or "macro_f1" not in df:
        return
    df = df[df["macro_f1"] != ""].copy()
    if df.empty:
        return
    df["macro_f1"] = df["macro_f1"].astype(float)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.bar(df["task"], df["macro_f1"], color="#4c78a8")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, max(1.0, float(df["macro_f1"].max()) * 1.1))
    ax.set_title("Paper pipeline smoke metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_design(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Paper Pipeline Experiment Design

## Claim To Support

Structured low-bandwidth interfaces can compose independently trained PSG experts while preserving modularity. The paper needs evidence against four alternatives: independent experts, naive late fusion/stacking, high-bandwidth fusion, and monolithic/shared multitask training.

## Datasets

- HSP: main training dataset with a deterministic subject-level split.
- HSP-test: held-out subjects from the same split for same-lab evaluation.
- RuhrlandKlinik: primary external-lab evaluation where task labels/channels exist.
- Other datasets: task-specific external checks later, depending on labels and channels.

## First Expert Suite

- Sleep staging: SleepTransformer on grouped EEG.
- Arousal: U-Time on EEG/EOG/chin EMG, using only EDF files with mapped arousal labels.
- Breathing: U-Time on abdomen/chest/airflow/SpO2.
- Desaturation: U-Time on SpO2.

## Method Grid

- Independent experts: each package evaluated alone.
- Stacking: frozen expert outputs/logits to a small task head.
- High-bandwidth fusion: existing MetaModel-style concatenation/fusion.
- Structured interfaces: directed low-bandwidth communication edges.
- Partial adaptation: frozen experts plus trainable interfaces/adapters.
- Monolithic multitask: one shared multitask model, used as a conventional baseline.

## Smoke Milestone

This milestone trains tiny one-epoch HSP experts and writes paper artifacts. Numbers are not scientific. A successful smoke means: split exists, package reload works, test JSONL exists, CSV/LaTeX/plot generation works, and every later full run has a stable place in the pipeline.
""",
        encoding="utf-8",
    )


def command_make_split(args) -> None:
    cfg = read_config(args.config)
    split = build_hsp_split(cfg)
    output_dir = Path(cfg["output_dir"])
    write_json(output_dir / "hsp_split.json", split)
    print(json.dumps({task: split["tasks"][task]["counts"] for task in PAPER_TASKS}, indent=2, sort_keys=True))


def command_train_smoke(args) -> None:
    cfg = read_config(args.config)
    output_dir = Path(cfg["output_dir"])
    split_path = Path(args.split) if args.split else output_dir / "hsp_split.json"
    if not split_path.exists():
        split = build_hsp_split(cfg)
        write_json(split_path, split)
    else:
        split = load_split(split_path)
    summaries = train_smoke(split, cfg, output_dir=output_dir)
    write_json(output_dir / "run_summaries.json", summaries)
    rows = metric_rows(summaries)
    write_csv(output_dir / "paper_smoke_results.csv", rows)
    write_latex_table(REPO_ROOT / "paper" / "tables" / "pipeline_smoke_results.tex", rows)
    write_metric_plot(REPO_ROOT / "paper" / "figures" / "pipeline_smoke_macro_f1.png", rows)


def command_collect(args) -> None:
    cfg = read_config(args.config)
    output_dir = Path(cfg["output_dir"])
    summaries = json.loads((output_dir / "run_summaries.json").read_text(encoding="utf-8"))
    rows = metric_rows(summaries)
    write_csv(output_dir / "paper_smoke_results.csv", rows)
    write_latex_table(REPO_ROOT / "paper" / "tables" / "pipeline_smoke_results.tex", rows)
    write_metric_plot(REPO_ROOT / "paper" / "figures" / "pipeline_smoke_macro_f1.png", rows)


def command_design(args) -> None:
    write_design(REPO_ROOT / "rebo" / "paper-pipeline.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_pipeline_smoke.yml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("design")
    subparsers.add_parser("make-split")
    train_parser = subparsers.add_parser("train-smoke")
    train_parser.add_argument("--split", default=None)
    subparsers.add_parser("collect")

    args = parser.parse_args()
    if args.command == "design":
        command_design(args)
    elif args.command == "make-split":
        command_make_split(args)
    elif args.command == "train-smoke":
        command_train_smoke(args)
    elif args.command == "collect":
        command_collect(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
