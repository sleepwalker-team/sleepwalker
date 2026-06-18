#!/bin/env python3

from __future__ import annotations

import argparse
from functools import partial

import numpy as np
import pandas as pd

from sleepwalker.datasets import Ruhrlandklinik
from sleepwalker.datasets.utils import get_edf_files_in_repo
from sleepwalker.deployment import load_models, predict_dataset
from sleepwalker.trainer.MultiLabelTrainer import MultiLabelTrainer
from sleepwalker.trainer.utils.metrics import cohen_kappa_from_confusion_matrix, f1_score_from_confusion_matrix
from sleepwalker.trainer.utils.targets import prepare_multiclass_target
from sleepwalker.utils import logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate exported .swmodel packages on a Ruhrland cohort.")
    parser.add_argument("--model", action="append", required=True, help="Model spec as path.swmodel or name=path.swmodel.")
    parser.add_argument("--data", required=True, help="Folder containing Ruhrland EDF files.")
    parser.add_argument("--device", default="cpu", help="Torch device used for inference.")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument("--num-workers-dataset", type=int, default=1, help="Workers used for Ruhrland dataset initialization.")
    parser.add_argument("--num-workers-loader", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--nox", action="store_true", help="Load Ruhrland NOX sidecar labels as target_extra.")
    parser.add_argument(
        "--target-resolution",
        action="append",
        default=[],
        help="Optional multiclass target resolution override as model=10s.",
    )
    parser.add_argument(
        "--depends",
        action="append",
        default=[],
        help="Optional dependency rule as child_ref=parent_ref:label1,label2 where refs are model or model.task.",
    )
    parser.add_argument("--show-confusion", action="store_true", help="Print confusion matrices.")
    return parser


def parse_key_value_list(items: list[str]) -> dict[str, str]:
    parsed = {}
    for item in items:
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def parse_dependencies(items: list[str]) -> list[dict[str, object]]:
    rules = []
    for item in items:
        child_ref, rhs = item.split("=", 1)
        parent_ref, labels = rhs.split(":", 1)
        rules.append(
            {
                "child": child_ref.strip(),
                "parent": parent_ref.strip(),
                "labels": [label.strip() for label in labels.split(",") if label.strip()],
            }
        )
    return rules


def resolve_target_resolution(loaded_model, overrides: dict[str, str]) -> pd.Timedelta:
    trainer = loaded_model.package.trainer
    if hasattr(trainer, "task_config"):
        return pd.to_timedelta(trainer.largest_task_resolution)

    if loaded_model.name in overrides:
        return pd.to_timedelta(overrides[loaded_model.name])

    metadata = loaded_model.package.metadata
    if "source_target_resolution" in metadata:
        return pd.to_timedelta(metadata["source_target_resolution"])

    # Guessing is unfortunately necessary here for multiclass models. The export
    # keeps an UnlabelledDataset for true deployment, and that object no longer
    # carries the original labeled target resolution. Until export stores this
    # explicitly, the best generic fallback is the unlabelled total_input.
    fallback = pd.to_timedelta(loaded_model.package.unlabelled_dataset.total_input)
    logger.warning(
        f"Guessing target_resolution={fallback} for multiclass model '{loaded_model.name}'. "
        "Pass --target-resolution or export this metadata explicitly if this is wrong."
    )
    return fallback


def build_ruhrland_dataset(loaded_model, data_path: str, use_nox: bool, overrides: dict[str, str], num_workers_dataset: int):
    package = loaded_model.package
    dataset_template = package.unlabelled_dataset
    trainer = package.trainer
    event_mapping = dict(package.metadata.get("source_event_mapping", {}) or {})

    if hasattr(trainer, "task_config"):
        prepare_target = partial(MultiLabelTrainer.prepare_target, task_config=trainer.task_config)
        target_resolution = pd.to_timedelta(trainer.largest_task_resolution)
    else:
        prepare_target = partial(prepare_multiclass_target, target_classes=list(trainer.classes))
        target_resolution = resolve_target_resolution(loaded_model, overrides)

    dataset = Ruhrlandklinik(
        channels=dataset_template.channels,
        sample_frequency=dataset_template.sample_frequency,
        resample_type=dataset_template.resample_type,
        total_input=dataset_template.total_input,
        target_resolution=target_resolution,
        stride=dataset_template.stride,
        event_mapping=event_mapping,
        prepare_patient=dataset_template.prepare_patient_callback,
        prepare_target=prepare_target,
        prepare_sample=dataset_template.prepare_sample_callback,
        online_max_tries=dataset_template.online_max_tries,
        force_one_day=dataset_template.force_one_day,
        rereference=dataset_template.rereference,
        return_nox=use_nox,
    )

    patients = sorted(get_edf_files_in_repo(data_path, recursive=True))
    if len(patients) == 0:
        raise ValueError(f"No EDF files found under {data_path}.")
    dataset.initialize(patients, num_workers=num_workers_dataset)
    return dataset


def extract_task_outputs(loaded_model, prediction_frame: pd.DataFrame, target_resolution: pd.Timedelta) -> dict[str, dict[str, object]]:
    trainer = loaded_model.package.trainer
    outputs: dict[str, dict[str, object]] = {}

    if hasattr(trainer, "task_config"):
        for task, cfg in trainer.task_config.items():
            rows = []
            for row in prediction_frame.to_dict(orient="records"):
                for step_idx in range(cfg["n_steps"]):
                    suffix = "" if cfg["n_steps"] == 1 else f"__step_{step_idx}"
                    rows.append(
                        {
                            "patient": row["patient"],
                            "time": pd.Timestamp(row[f"{task}{suffix}__time"]),
                            "pred": row[f"{task}{suffix}__prediction"],
                            "target": row.get(f"{task}{suffix}__target"),
                            "target_extra": row.get(f"{task}{suffix}__target_extra"),
                        }
                    )
            outputs[f"{loaded_model.name}.{task}"] = {
                "labels": list(cfg["labels"]),
                "resolution": pd.to_timedelta(cfg["target_resolution"]),
                "frame": pd.DataFrame(rows).dropna(subset=["time"]).sort_values(["patient", "time"]).reset_index(drop=True),
            }
        return outputs

    outputs[loaded_model.name] = {
        "labels": list(trainer.classes),
        "resolution": pd.to_timedelta(target_resolution),
        "frame": prediction_frame.rename(
            columns={
                "prediction": "pred",
                "target": "target",
                "target_extra": "target_extra",
            }
        )[["patient", "time", "pred", "target", "target_extra"]].copy(),
    }
    outputs[loaded_model.name]["frame"]["time"] = pd.to_datetime(outputs[loaded_model.name]["frame"]["time"])
    return outputs


def attach_parent_predictions(child_frame: pd.DataFrame, parent_frame: pd.DataFrame, parent_resolution: pd.Timedelta) -> pd.DataFrame:
    child_rows = []
    tolerance = parent_resolution / 2

    for patient, group in child_frame.groupby("patient", sort=False):
        child_patient = group.sort_values("time")
        parent_patient = parent_frame[parent_frame["patient"] == patient].sort_values("time")
        if len(parent_patient) == 0:
            continue

        merged = pd.merge_asof(
            child_patient,
            parent_patient[["time", "pred"]].rename(columns={"pred": "parent_pred"}),
            on="time",
            direction="nearest",
            tolerance=tolerance,
        )
        merged["patient"] = patient
        child_rows.append(merged)

    if len(child_rows) == 0:
        return pd.DataFrame(columns=list(child_frame.columns) + ["parent_pred"])
    return pd.concat(child_rows, ignore_index=True)


def apply_dependencies(task_ref: str, task_output: dict[str, object], dependency_rules: list[dict[str, object]], outputs_by_ref: dict[str, dict[str, object]]) -> dict[str, object]:
    frame = task_output["frame"].copy()
    for rule in dependency_rules:
        if rule["child"] != task_ref:
            continue
        parent_output = outputs_by_ref[rule["parent"]]
        frame = attach_parent_predictions(frame, parent_output["frame"], parent_output["resolution"])
        frame = frame[frame["parent_pred"].isin(rule["labels"])].drop(columns=["parent_pred"])
    out = dict(task_output)
    out["frame"] = frame.reset_index(drop=True)
    return out


def confusion_matrix_from_frame(frame: pd.DataFrame, labels: list[str], pred_column: str) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for row in frame.itertuples(index=False):
        truth = getattr(row, "target")
        pred = getattr(row, pred_column)
        if pd.isna(truth) or pd.isna(pred):
            continue
        cm[label_to_idx[truth], label_to_idx[pred]] += 1
    return cm


def metrics_from_confusion_matrix(cm: np.ndarray) -> dict[str, float]:
    total = int(cm.sum())
    accuracy = float(cm.trace() / total) if total > 0 else 0.0
    return {
        "n": total,
        "accuracy": accuracy,
        "f1_micro": f1_score_from_confusion_matrix(cm, macro=False) if total > 0 else 0.0,
        "f1_macro": f1_score_from_confusion_matrix(cm, macro=True) if total > 0 else 0.0,
        "kappa": cohen_kappa_from_confusion_matrix(cm) if total > 0 else 0.0,
    }


def format_confusion_matrix(cm: np.ndarray, labels: list[str]) -> str:
    return pd.DataFrame(cm, index=labels, columns=labels).to_string()


def main() -> None:
    args = build_parser().parse_args()
    target_resolution_overrides = parse_key_value_list(args.target_resolution)
    dependency_rules = parse_dependencies(args.depends)

    loaded_models = load_models(args.model, args.device)
    outputs_by_ref: dict[str, dict[str, object]] = {}
    summary_rows = []
    confusion_tables = []

    for loaded_model in loaded_models:
        dataset = build_ruhrland_dataset(
            loaded_model,
            args.data,
            use_nox=args.nox,
            overrides=target_resolution_overrides,
            num_workers_dataset=args.num_workers_dataset,
        )
        target_resolution = resolve_target_resolution(loaded_model, target_resolution_overrides)
        prediction_frame = predict_dataset(
            loaded_model,
            dataset,
            batch_size=args.batch_size,
            num_workers_loader=args.num_workers_loader,
            include_targets=True,
        )
        task_outputs = extract_task_outputs(loaded_model, prediction_frame, target_resolution)

        for task_ref, task_output in task_outputs.items():
            task_output = apply_dependencies(task_ref, task_output, dependency_rules, outputs_by_ref)
            outputs_by_ref[task_ref] = task_output

            model_cm = confusion_matrix_from_frame(task_output["frame"], task_output["labels"], "pred")
            model_metrics = metrics_from_confusion_matrix(model_cm)
            summary_rows.append({"task": task_ref, "source": "model", **model_metrics})

            if args.show_confusion:
                confusion_tables.append(f"\n[{task_ref}] model\n{format_confusion_matrix(model_cm, task_output['labels'])}")

            if args.nox and "target_extra" in task_output["frame"].columns:
                nox_frame = task_output["frame"].dropna(subset=["target_extra"])
                if len(nox_frame) > 0:
                    nox_cm = confusion_matrix_from_frame(nox_frame, task_output["labels"], "target_extra")
                    nox_metrics = metrics_from_confusion_matrix(nox_cm)
                    summary_rows.append({"task": task_ref, "source": "nox", **nox_metrics})
                    if args.show_confusion:
                        confusion_tables.append(f"\n[{task_ref}] nox\n{format_confusion_matrix(nox_cm, task_output['labels'])}")

    summary = pd.DataFrame(summary_rows)
    if len(summary) == 0:
        print("No evaluation rows produced.")
        return

    for metric in ["accuracy", "f1_micro", "f1_macro", "kappa"]:
        summary[metric] = summary[metric].map(lambda value: f"{float(value):.4f}")
    print(summary.to_string(index=False))

    if args.show_confusion and len(confusion_tables) > 0:
        print("\n".join(confusion_tables))


if __name__ == "__main__":
    main()
