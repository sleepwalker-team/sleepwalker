"""Prediction-table normalization used by analysis and deployment scripts."""

from __future__ import annotations

import re

import pandas as pd


def _sanitize_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return name or "model"


def _empty_records() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "group", "label", "prob"])


def _multiclass_records(frame: pd.DataFrame, model_key: str, classes: list[str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_records()
    probability_columns = [f"prob__{label}" for label in classes]
    missing = [column for column in ["time", *probability_columns] if column not in frame]
    if missing:
        raise ValueError(f"Prediction frame for '{model_key}' is missing {missing}.")
    records = frame[["time", *probability_columns]].copy()
    records["time"] = pd.to_datetime(records["time"])
    records = records.melt(id_vars="time", var_name="label", value_name="prob")
    records["label"] = records["label"].str.removeprefix("prob__")
    records["group"] = model_key
    return records[["time", "group", "label", "prob"]]


def _multilabel_records(frame: pd.DataFrame, model_key: str, task_config) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_records()
    task_records = []
    for task, config in task_config.items():
        labels = list(config["classes"])
        n_steps = int(config["n_steps"])
        for step in range(n_steps):
            suffix = "" if n_steps == 1 else f"__step_{step}"
            time_column = f"{task}{suffix}__time"
            prefix = f"{task}{suffix}__prob__"
            probability_columns = [f"{prefix}{class_name}" for class_name in labels]
            missing = [column for column in [time_column, *probability_columns] if column not in frame]
            if missing:
                raise ValueError(f"Prediction frame for task '{task}' is missing {missing}.")
            records = frame[[time_column, *probability_columns]].rename(
                columns={time_column: "time"}
            )
            records["time"] = pd.to_datetime(records["time"])
            records = records.melt(id_vars="time", var_name="label", value_name="prob")
            records["label"] = records["label"].str.removeprefix(prefix)
            records["group"] = f"{model_key}__{_sanitize_name(task)}"
            task_records.append(records[["time", "group", "label", "prob"]])
    return pd.concat(task_records, ignore_index=True) if task_records else _empty_records()


def prediction_frame_to_probability_records(
    frame: pd.DataFrame,
    model_key: str,
    classification_contract,
) -> pd.DataFrame:
    """Convert contract-specific wide predictions to one long probability schema."""
    contract_type = classification_contract.get("type")
    if contract_type == "multitask":
        return _multilabel_records(frame, model_key, classification_contract["tasks"])
    if contract_type == "single-head-multiclass":
        return _multiclass_records(frame, model_key, list(classification_contract["classes"]))
    raise ValueError(f"Unsupported classification contract type {contract_type!r}.")


def probability_records_to_onehot(records: pd.DataFrame) -> pd.DataFrame:
    """Average duplicate timestamps and select one winning label per model/task."""
    if records is None or records.empty:
        return pd.DataFrame()
    values = records.copy()
    values["time"] = pd.to_datetime(values["time"])
    values["prob"] = values["prob"].astype(float)
    values = (
        values.groupby(["time", "group", "label"], as_index=False)["prob"]
        .mean()
        .sort_values(
            ["time", "group", "prob", "label"],
            ascending=[True, True, False, True],
        )
    )
    winners = values.drop_duplicates(["time", "group"]).rename(
        columns={"label": "winner"}
    )[["time", "group", "winner"]]
    values = values.merge(winners, on=["time", "group"], how="left")
    values["value"] = (values["label"] == values["winner"]).astype("Int64")
    values["column"] = values["group"] + "__" + values["label"]
    wide = values.pivot(index="time", columns="column", values="value").sort_index()
    wide.index.name = "time"
    wide.columns.name = None
    return wide.sort_index(axis=1)
