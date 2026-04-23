#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Input, Static
from textual_plotext import PlotextPlot


SUPPORTED_MODES = ("train", "val", "test")
PLOT_METRICS = ("loss", "accuracy", "f1_macro", "coehns_kappa")

SORT_OPTIONS = [
    ("runid", "Run ID"),
    ("repeat", "Repeat"),
    ("status", "Status"),
    ("train_loss", "TRAIN loss"),
    ("train_accuracy", "TRAIN acc"),
    ("train_f1_macro", "TRAIN F1"),
    ("train_kappa", "TRAIN kappa"),
    ("val_loss", "VAL loss"),
    ("val_accuracy", "VAL acc"),
    ("val_f1_macro", "VAL F1"),
    ("val_kappa", "VAL kappa"),
    ("test_loss", "TEST loss"),
    ("test_accuracy", "TEST acc"),
    ("test_f1_macro", "TEST F1"),
    ("test_kappa", "TEST kappa"),
]

COLUMN_SPECS = [
    ("runid", "RUN ID", 14),
    ("repeat", "REPEAT", 8),
    ("status", "STATUS", 10),
    ("train_loss", "TRAIN loss", 11),
    ("train_accuracy", "TRAIN acc", 10),
    ("train_f1_macro", "TRAIN f1", 9),
    ("train_kappa", "TRAIN κ", 9),
    ("val_loss", "VAL loss", 10),
    ("val_accuracy", "VAL acc", 9),
    ("val_f1_macro", "VAL f1", 8),
    ("val_kappa", "VAL κ", 8),
    ("test_loss", "TEST loss", 10),
    ("test_accuracy", "TEST acc", 9),
    ("test_f1_macro", "TEST f1", 8),
    ("test_kappa", "TEST κ", 8),
]


@dataclass
class MlflowRun:
    run_id: str
    name: str
    status: str
    artifact_uri: str
    hparams_text: str
    hparams_lookup: dict[str, str]
    latest_metrics: dict[str, float | None]


@dataclass
class RunRecord:
    line_no: int
    name: str
    runid: str
    run_key: str
    model: str
    dataset: str
    repeat: str
    status: str
    folder_display: str
    hparams_text: str
    hparams_lookup: dict[str, str]
    classes: list[str]
    metrics: dict[str, float | None]
    confusion: dict[str, list[list[float]] | None]
    mlflow_run_id: str | None = None
    mlflow_artifact_uri: str | None = None


class MlflowSqlite:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.tables = self._get_tables()
        self.run_id_col = self._first_existing_column("runs", ("run_uuid", "run_id"))
        self.metric_run_id_col = self._first_existing_column("metrics", ("run_uuid", "run_id"))
        self.metric_value_col = self._first_existing_column("metrics", ("value",))
        self.metric_step_col = self._first_existing_column("metrics", ("step",))
        self.metric_ts_col = self._first_existing_column("metrics", ("timestamp",))

    def close(self) -> None:
        self.conn.close()

    def _get_tables(self) -> set[str]:
        rows = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return {str(row["name"]) for row in rows}

    def _columns(self, table: str) -> set[str]:
        if table not in self.tables:
            return set()
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row["name"]) for row in rows}

    def _first_existing_column(self, table: str, columns: tuple[str, ...]) -> str | None:
        existing = self._columns(table)
        for column in columns:
            if column in existing:
                return column
        return None

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.run_id_col or "runs" not in self.tables:
            return []
        rows = self.conn.execute(
            f"SELECT {self.run_id_col} AS run_id, "
            "COALESCE(name, '') AS name, "
            "COALESCE(status, '') AS status, "
            "COALESCE(artifact_uri, '') AS artifact_uri "
            "FROM runs "
            "WHERE COALESCE(lifecycle_stage, 'active') != 'deleted'"
        ).fetchall()
        return [dict(row) for row in rows]

    def latest_metrics_by_run(self) -> dict[str, dict[str, float | None]]:
        if not self.metric_run_id_col or not self.metric_value_col or "metrics" not in self.tables:
            return {}
        if "latest_metrics" in self.tables:
            latest_cols = self._columns("latest_metrics")
            latest_run_col = self._first_existing_column("latest_metrics", ("run_uuid", "run_id"))
            if latest_run_col and {"key", "value"} <= latest_cols:
                rows = self.conn.execute(
                    f"SELECT {latest_run_col} AS run_id, key, value FROM latest_metrics"
                ).fetchall()
                out: dict[str, dict[str, float | None]] = {}
                for row in rows:
                    out.setdefault(str(row["run_id"]), {})[str(row["key"])] = safe_float(row["value"])
                return out
        order_parts = []
        if self.metric_step_col:
            order_parts.append(f"{self.metric_step_col} ASC")
        if self.metric_ts_col:
            order_parts.append(f"{self.metric_ts_col} ASC")
        order_parts.append("rowid ASC")
        rows = self.conn.execute(
            f"SELECT {self.metric_run_id_col} AS run_id, key, {self.metric_value_col} AS value "
            f"FROM metrics ORDER BY {', '.join(order_parts)}"
        ).fetchall()
        out: dict[str, dict[str, float | None]] = {}
        for row in rows:
            out.setdefault(str(row["run_id"]), {})[str(row["key"])] = safe_float(row["value"])
        return out

    def fetch_metric_series_many(self, run_id: str, keys: list[str]) -> dict[str, list[tuple[int, float]]]:
        if not keys or not self.metric_run_id_col or not self.metric_value_col or "metrics" not in self.tables:
            return {}
        step_sql = self.metric_step_col if self.metric_step_col else "0"
        order_parts = []
        if self.metric_step_col:
            order_parts.append(f"{self.metric_step_col} ASC")
        if self.metric_ts_col:
            order_parts.append(f"{self.metric_ts_col} ASC")
        order_parts.append("rowid ASC")
        placeholders = ", ".join("?" for _ in keys)
        rows = self.conn.execute(
            f"SELECT key, {step_sql} AS step, {self.metric_value_col} AS value "
            f"FROM metrics WHERE {self.metric_run_id_col} = ? AND key IN ({placeholders}) "
            f"ORDER BY {', '.join(order_parts)}",
            [run_id, *keys],
        ).fetchall()
        out: dict[str, dict[int, float]] = {key: {} for key in keys}
        for row in rows:
            value = safe_float(row["value"])
            if value is None:
                continue
            out.setdefault(str(row["key"]), {})[int(row["step"] or 0)] = value
        return {
            key: sorted(step_map.items(), key=lambda item: item[0])
            for key, step_map in out.items()
            if step_map
        }


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_metric(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "-"
    if percent:
        return f"{100.0 * value:6.2f}"
    return f"{value:7.4f}"


def normalize_label(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 12 else text[:11] + "…"


def flatten_hparam_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(flatten_hparam_value(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key}={flatten_hparam_value(item)}" for key, item in value.items())
    return str(value)


def parse_hparams_text(text: str) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(text)
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        return {}, {}
    lookup = {str(key).strip().lower(): flatten_hparam_value(value).strip().lower() for key, value in parsed.items()}
    return parsed, lookup


def load_text(path: Path, missing_message: str) -> str:
    if not path.exists():
        return missing_message
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Could not read {path.name}: {exc}"


def resolve_run_folder(results_jsonl: Path, folder_value: Any) -> Path:
    candidate = Path(str(folder_value)) if folder_value not in (None, "") else Path(".")
    if candidate.is_absolute():
        return candidate
    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.exists():
        return cwd_path
    return (results_jsonl.parent / candidate).resolve()


def sanitize_cm(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or not value:
        return None
    rows: list[list[float]] = []
    width: int | None = None
    for row in value:
        if not isinstance(row, list) or not row:
            return None
        converted: list[float] = []
        for cell in row:
            number = safe_float(cell)
            if number is None:
                return None
            converted.append(number)
        width = len(converted) if width is None else width
        if len(converted) != width:
            return None
        rows.append(converted)
    if len(rows) != len(rows[0]):
        return None
    return rows


def metric_from_cm(cm: list[list[float]] | None) -> dict[str, float | None]:
    if not cm:
        return {"accuracy": None, "f1_macro": None, "kappa": None}
    total = float(sum(sum(row) for row in cm))
    if total <= 0:
        return {"accuracy": None, "f1_macro": None, "kappa": None}
    accuracy = sum(cm[idx][idx] for idx in range(len(cm))) / total
    per_class: list[float] = []
    for idx in range(len(cm)):
        tp = cm[idx][idx]
        fp = sum(cm[row][idx] for row in range(len(cm)) if row != idx)
        fn = sum(cm[idx][col] for col in range(len(cm)) if col != idx)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class.append(0.0 if precision == 0.0 and recall == 0.0 else 2.0 * precision * recall / (precision + recall))
    row_sums = [sum(row) for row in cm]
    col_sums = [sum(cm[row][col] for row in range(len(cm))) for col in range(len(cm))]
    expected = sum(row_sums[idx] * col_sums[idx] for idx in range(len(cm))) / (total * total)
    denominator = 1.0 - expected
    kappa = 0.0 if abs(denominator) < 1e-12 else (accuracy - expected) / denominator
    return {"accuracy": accuracy, "f1_macro": sum(per_class) / len(per_class), "kappa": kappa}


def extract_final_training_metrics(record: dict[str, Any]) -> tuple[dict[str, float | None], dict[str, list[list[float]] | None]]:
    metrics: dict[str, float | None] = {"train_loss": None, "val_loss": None}
    confusion: dict[str, list[list[float]] | None] = {"train": None, "val": None}
    train_loss = record.get("train_loss")
    if isinstance(train_loss, list) and train_loss and isinstance(train_loss[-1], dict):
        metrics["train_loss"] = safe_float(train_loss[-1].get("train"))
        metrics["val_loss"] = safe_float(train_loss[-1].get("val"))
    train_cm = record.get("train_cm")
    if isinstance(train_cm, list) and train_cm and isinstance(train_cm[-1], dict):
        confusion["train"] = sanitize_cm(train_cm[-1].get("train"))
        confusion["val"] = sanitize_cm(train_cm[-1].get("val"))
    return metrics, confusion


def metric_key(scope: str, mode: str, metric: str) -> str:
    return f"{scope}/{mode}/{metric}"


def latest_epoch_metric(latest_metrics: dict[str, float | None], mode: str, metric: str) -> float | None:
    return safe_float(latest_metrics.get(metric_key("epoch", mode, metric)))


def derive_run_key(name: str, hparams_lookup: dict[str, str]) -> str:
    run_id = hparams_lookup.get("id", "").strip()
    return run_id or name


def derive_display_id(name: str, repeat: Any) -> str:
    base = name.split("_")[-1] if "_" in name else name
    suffix = f"/r{repeat}" if repeat not in (None, "") else ""
    return f"{base}{suffix}"


def load_mlflow_runs(mlflow_path: Path) -> dict[str, MlflowRun]:
    backend = MlflowSqlite(mlflow_path)
    try:
        latest_by_run = backend.latest_metrics_by_run()
        runs: dict[str, MlflowRun] = {}
        for row in backend.list_runs():
            artifact_uri = str(row.get("artifact_uri") or "")
            hparams_path = Path(artifact_uri) / "hparams.json"
            hparams_text = load_text(hparams_path, "No hparams.json found for this MLflow run.")
            _, hparams_lookup = parse_hparams_text(hparams_text)
            run = MlflowRun(
                run_id=str(row["run_id"]),
                name=str(row["name"]),
                status=str(row["status"] or ""),
                artifact_uri=artifact_uri,
                hparams_text=hparams_text,
                hparams_lookup=hparams_lookup,
                latest_metrics=latest_by_run.get(str(row["run_id"]), {}),
            )
            run_key = derive_run_key(run.name, run.hparams_lookup)
            runs[run_key] = run
        return runs
    finally:
        backend.close()


def load_jsonl_runs(results_jsonl: Path) -> list[RunRecord]:
    runs: list[RunRecord] = []
    with results_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            training_metrics, training_confusion = extract_final_training_metrics(payload)
            test_cm = sanitize_cm(payload.get("test_cm"))
            train_metrics = metric_from_cm(training_confusion["train"])
            val_metrics = metric_from_cm(training_confusion["val"])
            test_metrics = metric_from_cm(test_cm)
            folder = resolve_run_folder(results_jsonl, payload.get("path"))
            hparams_text = load_text(folder / "hparams.yml", "No hparams.yml found for this run.")
            _, hparams_lookup = parse_hparams_text(hparams_text)
            name = str(payload.get("name", f"line-{line_no}"))
            repeat = payload.get("repeat", "")
            metrics = {
                "train_loss": training_metrics["train_loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_f1_macro": train_metrics["f1_macro"],
                "train_kappa": train_metrics["kappa"],
                "val_loss": training_metrics["val_loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_kappa": val_metrics["kappa"],
                "test_loss": safe_float(payload.get("test_loss")),
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_kappa": test_metrics["kappa"],
            }
            runs.append(
                RunRecord(
                    line_no=line_no,
                    name=name,
                    runid=derive_display_id(name, repeat),
                    run_key=derive_run_key(name, hparams_lookup),
                    model=str(payload.get("model", hparams_lookup.get("model", "-"))),
                    dataset=str(payload.get("dataset", "-")),
                    repeat=str(repeat) if repeat not in (None, "") else "-",
                    status="FINISHED",
                    folder_display=str(folder),
                    hparams_text=hparams_text,
                    hparams_lookup=hparams_lookup,
                    classes=[str(item) for item in payload.get("classes", [])],
                    metrics=metrics,
                    confusion={"train": training_confusion["train"], "val": training_confusion["val"], "test": test_cm},
                )
            )
    return runs


def build_runs(results_folder: Path) -> tuple[list[RunRecord], Path | None]:
    results_jsonl = results_folder / "results.jsonl"
    if not results_jsonl.exists():
        raise FileNotFoundError(f"results.jsonl not found in {results_folder}")
    mlflow_path = results_folder / "mlflow.sqlite"
    mlflow_runs = load_mlflow_runs(mlflow_path) if mlflow_path.exists() else {}

    finished_runs = load_jsonl_runs(results_jsonl)
    used_mlflow_keys: set[str] = set()
    merged: list[RunRecord] = []

    for run in finished_runs:
        mlflow = mlflow_runs.get(run.run_key)
        if mlflow:
            used_mlflow_keys.add(run.run_key)
            run.mlflow_run_id = mlflow.run_id
            run.mlflow_artifact_uri = mlflow.artifact_uri
        merged.append(run)

    for run_key, mlflow in mlflow_runs.items():
        if run_key in used_mlflow_keys and (mlflow.status or "").upper() == "FINISHED":
            continue
        metrics = {
            "train_loss": latest_epoch_metric(mlflow.latest_metrics, "train", "loss"),
            "train_accuracy": latest_epoch_metric(mlflow.latest_metrics, "train", "accuracy"),
            "train_f1_macro": latest_epoch_metric(mlflow.latest_metrics, "train", "f1_macro"),
            "train_kappa": latest_epoch_metric(mlflow.latest_metrics, "train", "coehns_kappa"),
            "val_loss": latest_epoch_metric(mlflow.latest_metrics, "val", "loss"),
            "val_accuracy": latest_epoch_metric(mlflow.latest_metrics, "val", "accuracy"),
            "val_f1_macro": latest_epoch_metric(mlflow.latest_metrics, "val", "f1_macro"),
            "val_kappa": latest_epoch_metric(mlflow.latest_metrics, "val", "coehns_kappa"),
            "test_loss": latest_epoch_metric(mlflow.latest_metrics, "test", "loss"),
            "test_accuracy": latest_epoch_metric(mlflow.latest_metrics, "test", "accuracy"),
            "test_f1_macro": latest_epoch_metric(mlflow.latest_metrics, "test", "f1_macro"),
            "test_kappa": latest_epoch_metric(mlflow.latest_metrics, "test", "coehns_kappa"),
        }
        merged.append(
            RunRecord(
                line_no=0,
                name=mlflow.name,
                runid=derive_display_id(mlflow.name, ""),
                run_key=run_key,
                model=mlflow.hparams_lookup.get("model", "-"),
                dataset="-",
                repeat="-",
                status=mlflow.status or "UNKNOWN",
                folder_display=mlflow.artifact_uri,
                hparams_text=mlflow.hparams_text,
                hparams_lookup=mlflow.hparams_lookup,
                classes=[],
                metrics=metrics,
                confusion={"train": None, "val": None, "test": None},
                mlflow_run_id=mlflow.run_id,
                mlflow_artifact_uri=mlflow.artifact_uri,
            )
        )

    return merged, mlflow_path if mlflow_path.exists() else None


def run_search_lookup(run: RunRecord) -> dict[str, str]:
    lookup = {
        "runid": run.runid.lower(),
        "id": run.name.lower(),
        "name": run.name.lower(),
        "model": run.model.lower(),
        "dataset": run.dataset.lower(),
        "repeat": run.repeat.lower(),
        "status": run.status.lower(),
        "path": run.folder_display.lower(),
        "folder": run.folder_display.lower(),
        "hparams": run.hparams_text.lower(),
    }
    lookup.update(run.hparams_lookup)
    return lookup


def token_matches_run(run: RunRecord, token: str) -> bool:
    lookup = run_search_lookup(run)
    token = token.strip().lower()
    if not token:
        return True
    for separator in ("=", ":"):
        if separator in token:
            key, value = token.split(separator, 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                return token in " ".join(lookup.values())
            field_value = lookup.get(key)
            return field_value is not None and value in field_value
    return token in " ".join(lookup.values())


def sort_value(run: RunRecord, key: str) -> Any:
    if key == "runid":
        return run.runid.lower()
    if key == "repeat":
        repeat = safe_float(run.repeat)
        return repeat if repeat is not None else run.repeat.lower()
    if key == "status":
        order = {"RUNNING": 0, "SCHEDULED": 1, "FINISHED": 2, "FAILED": 3, "KILLED": 4}
        return order.get(run.status.upper(), 99)
    value = run.metrics.get(key)
    return -math.inf if value is None else value


def filter_runs(runs: list[RunRecord], query: str, sort_key: str, descending: bool) -> list[RunRecord]:
    tokens = [token for token in query.strip().lower().split() if token]
    filtered = []
    for run in runs:
        if tokens and not all(token_matches_run(run, token) for token in tokens):
            continue
        filtered.append(run)
    return sorted(filtered, key=lambda item: sort_value(item, sort_key), reverse=descending)


def cell_style(value: float, max_value: float) -> str:
    if max_value <= 0:
        return "white"
    ratio = 0.15 + 0.85 * (value / max_value)
    red = int(120 + 35 * ratio)
    green = int(145 + 55 * ratio)
    blue = int(170 + 70 * ratio)
    return f"bold rgb({red},{green},{blue})"


def render_confusion_panel(run: RunRecord | None, scope: str) -> Panel:
    title_scope = scope.upper()
    if run is None:
        return Panel("No run selected.", title=title_scope, padding=(0, 1), box=box.ROUNDED, border_style="#4c566a")
    if run.status.upper() != "FINISHED":
        return Panel("Confusion matrices are only shown for finished runs.", title=title_scope, padding=(0, 1), box=box.ROUNDED, border_style="#4c566a")
    cm = run.confusion.get(scope)
    if not cm:
        return Panel("No confusion matrix available.", title=title_scope, padding=(0, 1), box=box.ROUNDED, border_style="#4c566a")

    labels = run.classes or [str(idx) for idx in range(len(cm))]
    total = sum(sum(row) for row in cm)
    max_value = max(max(row) for row in cm) if cm else 0.0
    summary = Text()
    summary.append(f"loss {format_metric(run.metrics.get(f'{scope}_loss'))}    ")
    summary.append(f"acc {format_metric(run.metrics.get(f'{scope}_accuracy'), percent=True)}    ")
    summary.append(f"f1 {format_metric(run.metrics.get(f'{scope}_f1_macro'))}    ")
    summary.append(f"κ {format_metric(run.metrics.get(f'{scope}_kappa'))}")
    table = Table(expand=True, show_header=True, header_style="bold")
    table.add_column("actual \\ pred", justify="left", style="bold")
    for label in labels:
        table.add_column(normalize_label(label), justify="center")
    for row_idx, row in enumerate(cm):
        row_total = sum(row)
        cells: list[Text] = []
        for value in row:
            pct = (value / row_total) if row_total > 0 else 0.0
            cell = Text(justify="center")
            cell.append(f"{int(value):,}\n", style=cell_style(value, max_value))
            cell.append(f"{pct:5.1%}", style="dim")
            cells.append(cell)
        table.add_row(normalize_label(labels[row_idx]), *cells)
        if row_idx < len(cm) - 1:
            table.add_row("", *([""] * len(labels)))
    footer = Text(f"support {int(total):,} samples", style="dim")
    return Panel(Group(summary, table, footer), title=title_scope, padding=(0, 1), box=box.ROUNDED, border_style="#4c566a")


def render_run_summary(run: RunRecord | None) -> Panel:
    if run is None:
        return Panel("No runs match the current search.", title="Selection", padding=(0, 1), box=box.MINIMAL, border_style="#4c566a")
    body = Text()
    body.append(f"{run.name}\n", style="bold")
    body.append(f"status={run.status}    repeat={run.repeat}    model={run.model}\n")
    body.append("\nHPARAMS\n", style="bold")
    body.append(run.hparams_text, style="dim")
    return Panel(body, title="Selection", padding=(0, 1), box=box.MINIMAL, border_style="#4c566a")


def smooth_series(series: list[tuple[int, float]], window: int = 25) -> list[tuple[int, float]]:
    if window <= 1 or len(series) <= 2:
        return list(series)
    out: list[tuple[int, float]] = []
    values: list[float] = []
    for step, value in series:
        values.append(value)
        current = values[max(0, len(values) - window) :]
        out.append((step, sum(current) / len(current)))
    return out


def downsample_series(series: list[tuple[int, float]], max_points: int) -> list[tuple[int, float]]:
    if max_points <= 0 or len(series) <= max_points:
        return list(series)
    stride = max(1, math.ceil(len(series) / max_points))
    sampled = series[::stride]
    if sampled[-1] != series[-1]:
        sampled.append(series[-1])
    return sampled


def prepare_series_for_plot(scope: str, series: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not series:
        return []
    if scope == "batch":
        series = smooth_series(series[-3000:], window=25)
        return downsample_series(series, max_points=420)
    return downsample_series(series[-800:], max_points=420)


class DetailsScreen(ModalScreen[None]):
    CSS = """
    DetailsScreen {
        align: center middle;
        background: rgba(46, 52, 64, 0.78);
    }

    #details-shell {
        width: 96%;
        height: 94%;
        border: round #4c566a;
        padding: 1;
        background: #2e3440;
    }

    #details-title {
        height: auto;
        content-align: center middle;
        text-style: bold;
        margin-bottom: 1;
        color: #eceff4;
    }

    #details-subtitle {
        height: auto;
        margin-bottom: 1;
        color: #d8dee9;
    }

    #details-grid {
        grid-size: 2 4;
        grid-columns: 1fr 1fr;
        grid-rows: 1fr 1fr 1fr 1fr;
        grid-gutter: 1 2;
        height: 1fr;
    }

    .plot-card {
        border: round #4c566a;
        padding: 0 1;
        background: #3b4252;
    }

    .plot-title {
        height: auto;
        text-style: bold;
        margin-bottom: 0;
        color: #eceff4;
    }

    .metric-plot {
        height: 1fr;
    }
    """

    BINDINGS = [Binding("escape", "dismiss", "Close"), Binding("q", "dismiss", "Close")]

    def __init__(self, run: RunRecord, metric_series: dict[str, list[tuple[int, float]]]) -> None:
        super().__init__()
        self.run = run
        self._series_cache = metric_series

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(f"{self.run.name}  |  {self.run.status}", id="details-title"),
            Static(self.run.folder_display, id="details-subtitle"),
            Grid(
                *[
                    Vertical(
                        Static(f"{metric.replace('_', ' ')}  |  {scope}", classes="plot-title"),
                        PlotextPlot(id=f"plot-{scope}-{metric}", classes="metric-plot"),
                        classes="plot-card",
                    )
                    for metric in ("loss", "accuracy", "f1_macro", "coehns_kappa")
                    for scope in ("batch", "epoch")
                ],
                id="details-grid",
            ),
            id="details-shell",
        )

    def on_mount(self) -> None:
        self.app.theme = "nord"
        self._refresh_plots()

    def _refresh_plots(self) -> None:
        colors = {"train": "green", "val": "yellow", "test": "cyan"}
        for metric in ("loss", "accuracy", "f1_macro", "coehns_kappa"):
            for scope in ("batch", "epoch"):
                plot = self.query_one(f"#plot-{scope}-{metric}", PlotextPlot)
                plt = plot.plt
                plt.clf()
                plt.theme("clear")
                plt.grid(False, False)
                plt.frame(True)
                plt.xaxes(True, False)
                plt.yaxes(True, False)
                plt.canvas_color("default")
                plt.axes_color("default")
                plt.ticks_color("default")
                plt.xfrequency(4)
                plt.yfrequency(4)
                has_any = False
                for mode in SUPPORTED_MODES:
                    key = metric_key(scope, mode, metric)
                    series = self._series_cache.get(key, [])
                    if not series:
                        continue
                    has_any = True
                    xs = list(range(len(series)))
                    ys = [value for _, value in series]
                    plt.plot(xs, ys, label=mode, color=colors[mode], marker="braille")
                if has_any:
                    plt.title(f"{scope}/{metric}")
                else:
                    plt.title("no mlflow data")
                plot.refresh()

    def action_dismiss(self) -> None:
        self.dismiss(None)


class ResultsFolderApp(App[None]):
    CSS = """
    #main {
        height: 1fr;
        layout: horizontal;
    }

    #left-pane {
        width: 2fr;
        min-width: 90;
        margin: 0 1 1 1;
    }

    #right-pane {
        width: 1fr;
        min-width: 36;
        margin: 0 1 1 0;
    }

    #filter {
        width: 1fr;
        height: auto;
        margin-bottom: 1;
        border: round #4c566a;
    }

    #runs-table {
        height: 1fr;
        border: round #4c566a;
    }

    #selection-scroll {
        height: 22;
        margin-bottom: 1;
        border: round #4c566a;
    }

    #selection-summary {
        height: auto;
    }

    #matrix-stack {
        height: 1fr;
    }

    .cm-view {
        height: auto;
        padding: 0;
        margin: 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("/", "focus_filter", "Search"),
        Binding("enter", "open_details", "Details"),
        Binding("s", "toggle_sort_direction", "Asc/Desc"),
    ]

    def __init__(self, results_folder: Path) -> None:
        super().__init__()
        self.theme = "nord"
        self.results_folder = results_folder
        self.all_runs, self.mlflow_path = build_runs(results_folder)
        self.mlflow_backend = MlflowSqlite(self.mlflow_path) if self.mlflow_path is not None else None
        self.metric_series_cache: dict[str, dict[str, list[tuple[int, float]]]] = {}
        self.plot_series_cache: dict[str, dict[str, list[tuple[int, float]]]] = {}
        self.visible_runs: list[RunRecord] = []
        self.sort_key = "test_kappa"
        self.sort_descending = True

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Vertical(
                Input(placeholder="Live search: repeat=2 model=utime-big grouped=true", id="filter"),
                DataTable(id="runs-table"),
                id="left-pane",
            ),
            Vertical(
                VerticalScroll(Static(id="selection-summary"), id="selection-scroll"),
                Vertical(
                    Static(id="cm-test", classes="cm-view"),
                    Static(id="cm-val", classes="cm-view"),
                    Static(id="cm-train", classes="cm-view"),
                    id="matrix-stack",
                ),
                id="right-pane",
            ),
            id="main",
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        for key, label, width in COLUMN_SPECS:
            table.add_column(label, key=key, width=width)
        self.refresh_table()
        table.focus()

    def on_unmount(self) -> None:
        if self.mlflow_backend is not None:
            self.mlflow_backend.close()
            self.mlflow_backend = None

    def selected_run(self) -> RunRecord | None:
        table = self.query_one("#runs-table", DataTable)
        row_index = table.cursor_row
        if row_index is None or row_index < 0 or row_index >= len(self.visible_runs):
            return None
        return self.visible_runs[row_index]

    def refresh_table(self) -> None:
        query = self.query_one("#filter", Input).value if self.is_mounted else ""
        self.visible_runs = filter_runs(self.all_runs, query, self.sort_key, self.sort_descending)
        table = self.query_one("#runs-table", DataTable)
        table.clear(columns=False)
        for run in self.visible_runs:
            table.add_row(
                run.runid,
                run.repeat,
                run.status,
                format_metric(run.metrics["train_loss"]),
                format_metric(run.metrics["train_accuracy"], percent=True),
                format_metric(run.metrics["train_f1_macro"]),
                format_metric(run.metrics["train_kappa"]),
                format_metric(run.metrics["val_loss"]),
                format_metric(run.metrics["val_accuracy"], percent=True),
                format_metric(run.metrics["val_f1_macro"]),
                format_metric(run.metrics["val_kappa"]),
                format_metric(run.metrics["test_loss"]),
                format_metric(run.metrics["test_accuracy"], percent=True),
                format_metric(run.metrics["test_f1_macro"]),
                format_metric(run.metrics["test_kappa"]),
            )
        if self.visible_runs:
            table.move_cursor(row=0, column=0)
            self.prefetch_plot_series_for_run(self.visible_runs[0])
            self.update_right_pane(self.visible_runs[0])
        else:
            self.update_right_pane(None)

    def update_right_pane(self, run: RunRecord | None) -> None:
        self.query_one("#selection-summary", Static).update(render_run_summary(run))
        self.query_one("#cm-test", Static).update(render_confusion_panel(run, "test"))
        self.query_one("#cm-val", Static).update(render_confusion_panel(run, "val"))
        self.query_one("#cm-train", Static).update(render_confusion_panel(run, "train"))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter":
            self.refresh_table()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "runs-table":
            run = self.selected_run()
            if run is not None:
                self.prefetch_plot_series_for_run(run)
            self.update_right_pane(run)

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        if event.data_table.id != "runs-table":
            return
        column_key = str(event.column_key.value)
        if column_key == self.sort_key:
            self.sort_descending = not self.sort_descending
        else:
            self.sort_key = column_key
            self.sort_descending = column_key not in {"runid", "repeat", "status"}
        self.refresh_table()

    def on_key(self, event: Key) -> None:
        if event.key == "enter":
            focused = self.focused
            if isinstance(focused, DataTable) and focused.id == "runs-table":
                run = self.selected_run()
                if run is not None:
                    event.stop()
                    self.push_screen(DetailsScreen(run, self.plot_series_for_run(run)))

    def action_open_details(self) -> None:
        run = self.selected_run()
        if run is not None:
            self.push_screen(DetailsScreen(run, self.plot_series_for_run(run)))

    def action_focus_filter(self) -> None:
        self.query_one("#filter", Input).focus()

    def action_toggle_sort_direction(self) -> None:
        self.sort_descending = not self.sort_descending
        self.refresh_table()

    def metric_series_for_run(self, run: RunRecord) -> dict[str, list[tuple[int, float]]]:
        if run.mlflow_run_id is None or self.mlflow_backend is None:
            return {}
        cached = self.metric_series_cache.get(run.mlflow_run_id)
        if cached is not None:
            return cached
        wanted_keys = [
            metric_key(scope, mode, metric)
            for metric in PLOT_METRICS
            for scope in ("batch", "epoch")
            for mode in SUPPORTED_MODES
        ]
        cached = self.mlflow_backend.fetch_metric_series_many(run.mlflow_run_id, wanted_keys)
        self.metric_series_cache[run.mlflow_run_id] = cached
        return cached

    def plot_series_for_run(self, run: RunRecord) -> dict[str, list[tuple[int, float]]]:
        if run.mlflow_run_id is None:
            return {}
        cached = self.plot_series_cache.get(run.mlflow_run_id)
        if cached is not None:
            return cached
        raw_series = self.metric_series_for_run(run)
        prepared = {
            key: prepare_series_for_plot("batch" if key.startswith("batch/") else "epoch", series)
            for key, series in raw_series.items()
        }
        self.plot_series_cache[run.mlflow_run_id] = prepared
        return prepared

    def prefetch_plot_series_for_run(self, run: RunRecord) -> None:
        if run.mlflow_run_id is None:
            return
        if run.mlflow_run_id in self.plot_series_cache:
            return
        self.plot_series_for_run(run)


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore a results folder containing results.jsonl and optional mlflow.sqlite.")
    parser.add_argument("results_folder", type=Path, help="Folder that contains results.jsonl")
    args = parser.parse_args()
    if not args.results_folder.exists():
        raise SystemExit(f"Results folder not found: {args.results_folder}")
    app = ResultsFolderApp(args.results_folder.resolve())
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
