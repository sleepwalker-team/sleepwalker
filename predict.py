"""Inference CLI for exported Sleepwalker prediction packages.

The script loads one or more ``.swmodel`` bundles, applies them to EDF files,
and writes one wide one-hot prediction table per EDF file. Unit tests currently
cover the prediction-frame reshaping helpers, but actual deployment workflows
should still be treated as evolving lab-internal tooling.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

import pandas as pd

from sleepwalker.deployment import load_prediction_package
from sleepwalker.utils import logger


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "model"


def _unique_names(paths: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    unique = []
    for path in paths:
        base = _sanitize_name(Path(path).stem)
        counts[base] += 1
        suffix = counts[base]
        unique.append(base if suffix == 1 else f"{base}_{suffix}")
    return unique


def _empty_probability_records() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "group", "label", "prob"])


def _multiclass_probability_records(frame: pd.DataFrame, model_key: str, classes: list[str]) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        return _empty_probability_records()

    prob_columns = [f"prob__{label}" for label in classes]
    missing = [col for col in ["time", *prob_columns] if col not in frame.columns]
    if len(missing) > 0:
        raise ValueError(
            f"Prediction frame for model '{model_key}' is missing multiclass columns: {missing}"
        )

    out = frame[["time", *prob_columns]].copy()
    out["time"] = pd.to_datetime(out["time"])
    out = out.melt(id_vars=["time"], var_name="label", value_name="prob")
    out["label"] = out["label"].str.replace(r"^prob__", "", regex=True)
    out["group"] = model_key
    return out[["time", "group", "label", "prob"]]


def _multilabel_probability_records(frame: pd.DataFrame, model_key: str, task_config) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        return _empty_probability_records()

    task_frames = []
    for task, cfg in task_config.items():
        task_group = f"{model_key}__{_sanitize_name(task)}"
        n_steps = int(cfg["n_steps"])
        labels = list(cfg["labels"])
        for step_idx in range(n_steps):
            suffix = "" if n_steps == 1 else f"__step_{step_idx}"
            time_col = f"{task}{suffix}__time"
            prob_columns = [f"{task}{suffix}__prob__{label}" for label in labels]
            missing = [col for col in [time_col, *prob_columns] if col not in frame.columns]
            if len(missing) > 0:
                raise ValueError(
                    f"Prediction frame for task '{task}' in model '{model_key}' is missing columns: {missing}"
                )

            task_frame = frame[[time_col, *prob_columns]].copy()
            task_frame = task_frame.rename(columns={time_col: "time"})
            task_frame["time"] = pd.to_datetime(task_frame["time"])
            task_frame = task_frame.melt(id_vars=["time"], var_name="label", value_name="prob")
            task_frame["label"] = task_frame["label"].str.replace(
                rf"^{re.escape(task + suffix + '__prob__')}", "", regex=True
            )
            task_frame["group"] = task_group
            task_frames.append(task_frame[["time", "group", "label", "prob"]])

    if len(task_frames) == 0:
        return _empty_probability_records()
    return pd.concat(task_frames, ignore_index=True)


def prediction_frame_to_probability_records(frame: pd.DataFrame, model_key: str, trainer) -> pd.DataFrame:
    """Normalize trainer-specific prediction frames into long probability rows.

    Args:
        frame: Prediction DataFrame returned by a trainer.
        model_key: Stable name assigned to the loaded model package.
        trainer: Trainer object that determines whether the frame is interpreted
            as multiclass or multitask output.

    Returns:
        A long-form DataFrame with columns ``time``, ``group``, ``label``, and
        ``prob``.

    Raises:
        ValueError: If the trainer type is unsupported or required columns are
            missing from the prediction frame.
    """
    if hasattr(trainer, "task_config"):
        return _multilabel_probability_records(frame, model_key, trainer.task_config)
    if hasattr(trainer, "classes"):
        return _multiclass_probability_records(frame, model_key, list(trainer.classes))
    raise ValueError(
        f"Unsupported trainer type {type(trainer).__name__}. Expected a trainer exposing classes or task_config."
    )


def probability_records_to_onehot(probability_records: pd.DataFrame) -> pd.DataFrame:
    """Convert long probability rows into a wide winner-take-all table.

    Args:
        probability_records: Long-form probability rows produced by
            :func:`prediction_frame_to_probability_records`.

    Returns:
        A time-indexed DataFrame with one column per ``group__label`` pair.
        Duplicate ``time/group/label`` records are averaged first, matching the
        behavior asserted in ``tests/test_predict.py``.
    """
    if probability_records is None or len(probability_records) == 0:
        return pd.DataFrame()

    records = probability_records.copy()
    records["time"] = pd.to_datetime(records["time"])
    records["prob"] = records["prob"].astype(float)
    grouped = (
        records.groupby(["time", "group", "label"], as_index=False)["prob"]
        .mean()
        .sort_values(["time", "group", "prob", "label"], ascending=[True, True, False, True])
    )
    winners = grouped.drop_duplicates(["time", "group"], keep="first")
    winners = winners.rename(columns={"label": "winner_label"})[["time", "group", "winner_label"]]
    onehot = grouped.merge(winners, on=["time", "group"], how="left")
    onehot["value"] = (onehot["label"] == onehot["winner_label"]).astype("Int64")
    onehot["column"] = onehot["group"] + "__" + onehot["label"]
    wide = onehot.pivot(index="time", columns="column", values="value").sort_index()
    wide.columns.name = None
    wide.index.name = "time"
    return wide.sort_index(axis=1)


def predict_edf(packages: list[tuple[str, object]], edf_path: str, batch_size: int, num_workers_dataset: int, num_workers_loader: int) -> pd.DataFrame:
    """Run all loaded packages on one EDF file and merge their outputs.

    Args:
        packages: ``(model_key, PredictionPackage)`` pairs.
        edf_path: EDF file to score.
        batch_size: Inference batch size.
        num_workers_dataset: Worker count used for dataset preparation.
        num_workers_loader: Worker count used by the prediction dataloader.

    Returns:
        A wide one-hot prediction table, or an empty DataFrame when no package
        produces predictions.
    """
    probability_frames = []
    for model_key, package in packages:
        logger.info(f"Applying model '{model_key}' to {edf_path}")
        frame = package.predict_patient(
            edf_path,
            batch_size=batch_size,
            num_workers_dataset=num_workers_dataset,
            num_workers_loader=num_workers_loader,
        )
        if frame is None or len(frame) == 0:
            logger.warning(f"Model '{model_key}' produced no predictions for {edf_path}")
            continue
        probability_frames.append(
            prediction_frame_to_probability_records(frame, model_key=model_key, trainer=package.trainer)
        )

    if len(probability_frames) == 0:
        logger.warning(f"No predictions were produced for {edf_path}")
        return pd.DataFrame()
    return probability_records_to_onehot(pd.concat(probability_frames, ignore_index=True))


def _output_path(output_dir: Path, edf_path: str, suffix: str, used_names: Counter[str]) -> Path:
    base = _sanitize_name(Path(edf_path).stem)
    used_names[base] += 1
    if used_names[base] > 1:
        base = f"{base}_{used_names[base]}"
    return output_dir / f"{base}{suffix}"


def _write_dataframe(df: pd.DataFrame, output_path: Path, output_format: str) -> None:
    if output_format == "csv":
        df.to_csv(output_path)
        return
    if output_format == "parquet":
        df.to_parquet(output_path)
        return
    raise ValueError(f"Unsupported output format '{output_format}'.")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the prediction CLI."""
    parser = argparse.ArgumentParser(description="Run one or more exported Sleepwalker models on EDF files.")
    parser.add_argument("--models", nargs="+", required=True, help="One or more .swmodel files.")
    parser.add_argument("--edf-files", nargs="+", required=True, help="One or more EDF files to predict.")
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"), help="Directory for per-EDF outputs.")
    parser.add_argument("--output-format", choices=["csv", "parquet"], default="csv", help="Output table format.")
    parser.add_argument("--device", default="cpu", help="Torch device used for inference.")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument("--num-workers-dataset", type=int, default=1, help="Dataset initialization workers.")
    parser.add_argument("--num-workers-loader", type=int, default=0, help="Dataloader workers.")
    return parser


def main() -> None:
    """Load exported packages, predict EDF files, and write output tables."""
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_keys = _unique_names(args.models)
    logger.progress_start(len(args.models), desc="Loading models", leave=True)
    packages = []
    for model_key, model_path in zip(model_keys, args.models):
        logger.info(f"Loading {model_path} as '{model_key}'")
        packages.append((model_key, load_prediction_package(model_path, map_location=args.device)))
        logger.progress_advance(1)
    logger.progress_close()

    suffix = ".csv" if args.output_format == "csv" else ".parquet"
    used_output_names: Counter[str] = Counter()
    logger.progress_start(len(args.edf_files), desc="Predicting EDF files", leave=True)
    for edf_path in args.edf_files:
        logger.context(Path(edf_path).name)
        df = predict_edf(
            packages,
            edf_path,
            batch_size=args.batch_size,
            num_workers_dataset=args.num_workers_dataset,
            num_workers_loader=args.num_workers_loader,
        )
        output_path = _output_path(output_dir, edf_path, suffix, used_output_names)
        _write_dataframe(df, output_path, args.output_format)
        logger.info(f"Wrote predictions to {output_path}")
        logger.uncontext()
        logger.progress_advance(1)
    logger.progress_close()


if __name__ == "__main__":
    main()
