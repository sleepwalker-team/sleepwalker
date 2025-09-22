from __future__ import annotations

import os
import shutil
import tempfile
from abc import ABC
from collections import defaultdict
from typing import Optional
import typing

import mlflow
import pandas as pd

from sleepwalker.trainer.logger.Logger import Logger

class MlflowLogger(Logger):
    """
    Minimal MLflow-backed implementation of the Logger ABC.

    Usage:
        logger = MlflowLogger("MyExperiment", "run-42", tracking_uri="file:./mlruns")
        logger.log_metric("loss", 0.123) 
        ...
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        tracking_uri: Optional[str] = None,
        tags: Optional[dict] = None,
        resume_if_exists: bool = False,
    ):
        super().__init__(experiment_name, run_name)
        self._tmpdir = tempfile.mkdtemp(prefix="mlflow_logger_")
        self._closed = False

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        # Ensure experiment exists and get id
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = exp.experiment_id

        # Optionally resume an existing (active) run with same name; otherwise start a new run
        active_run = mlflow.active_run()
        if active_run is not None:
            # Respect an existing outer run/context if user manages lifecycle
            self._run = active_run
        else:
            existing = None
            if resume_if_exists:
                # Best-effort: search for a run with same name in this experiment
                runs = mlflow.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string=f"tags.mlflow.runName = '{run_name}' and attributes.status = 'RUNNING'",
                    max_results=1,
                    output_format="pandas"
                )
                if len(runs) > 0:
                    runs = typing.cast(pd.DataFrame, runs)
                    existing = runs.iloc[0]

            self._run = mlflow.start_run(
                run_id=(existing.run_id if existing is not None else None),
                experiment_id=experiment_id,
                run_name=run_name,
                tags=tags or {},
            )

    def log_metric(self, metric_name, metric_value):
        mlflow.log_metric(metric_name, float(metric_value))

    def log_artifact(self, source_path, artifact_path):
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Artifact source not found: {source_path}")
        
        mlflow.log_artifact(local_path=source_path, artifact_path=artifact_path)
