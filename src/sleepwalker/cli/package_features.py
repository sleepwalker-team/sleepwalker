#!/usr/bin/env python3
"""Fit feature extractors on a training config's data and save them as an embedding-only package.

    sleepwalker package-features configs/examples/rocket_head_ruhrland.yml -f minirocket -o /raid/packages/minirocket [--dry]

The package takes channels, units, sample frequency and sleep trimming from the config's first ``data`` block.
Rocket biases are quantiles of the data, so the extractors are fitted here, on random windows from that block's
training split only. Train a head on the package with ``PackagedClassifierModel`` in the same config.
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Any, Mapping

from sleepwalker.cli.train import build_dataset, load_patient_split
from sleepwalker.config import apply_patient_filter, read_yaml
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.deployment import save_packaged_model
from sleepwalker.features import Featurizer
from sleepwalker.models.PackagedEmbeddingModel import PackagedEmbeddingModel
from sleepwalker.models.preprocessors.Featurize import Featurize
from sleepwalker.training.loader import build_loader
from sleepwalker.utils import logger

DRY_RUN_PATIENTS = 2


def one_epoch_entry(data_config: Mapping[str, Any], epoch: str) -> dict[str, Any]:
    """A copy of the data config with one-epoch windows and labels: the package's window, whatever context the head uses."""
    one_epoch = copy.deepcopy(dict(data_config))
    one_epoch["total_input"] = epoch
    one_epoch["stride"] = epoch
    if isinstance(one_epoch.get("prepare_target"), dict):
        one_epoch["prepare_target"]["target_resolution"] = epoch
        one_epoch["prepare_target"].pop("target_offset", None)  # a one-epoch window has no offset
        if "sequence_len" in one_epoch["prepare_target"]:
            one_epoch["prepare_target"]["sequence_len"] = 1
    return one_epoch


def warmup(featurize: Featurize, loader) -> None:
    """Fit the featurizer on loader windows, stopping at ``featurize.fit_windows``.

    A loader that runs out first still fits on what it saw.
    """
    if not featurize.requires_warmup():  # only stateless extractors: nothing to fit
        return
    seen = 0
    for batch in loader:
        if batch is None:  # every sample in the batch was rejected by prepare_target
            continue
        featurize.update(batch["data"])
        # Windows, not batches: a batch with rejected samples is shorter than batch_size.
        seen += len(batch["data"])
        if seen >= featurize.fit_windows:
            break
    featurize.fit_buffered()
    logger.info(f"Warmed up the featurizer on {seen} train window(s) (budget {featurize.fit_windows}).")
    if not featurize.fitted:
        raise RuntimeError("The featurizer is still unfitted after warmup; refusing to save a package that cannot transform.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sleepwalker package-features", description="Package feature extractors as an embedding model.")
    parser.add_argument("config", help="Training config; its first data block defines channels, units and the training split.")
    parser.add_argument("-f", "--features", nargs="+", required=True, help="Registered extractor names, e.g. minirocket.")
    parser.add_argument("-o", "--out", required=True, help="Package destination directory.")
    parser.add_argument("--epoch", default="30s", help="Native window of the package.")
    parser.add_argument("--fit-windows", type=int, default=1024, help="Warmup budget per stage, in windows.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-jobs", type=int, default=None, help="CPUs that featurize each batch wherever the package runs; stored in the package. Default 1, with a warning.")
    parser.add_argument("--num-workers", type=int, default=4, help="Warmup data-loading workers.")
    parser.add_argument("--dry", action="store_true", help=f"Use {DRY_RUN_PATIENTS} training patients, for a quick check.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.n_jobs is None:
        logger.warning("No --n-jobs given: the package featurizes on 1 CPU wherever it runs. Pass --n-jobs N to use more.")
        args.n_jobs = 1
    config = read_yaml(args.config)
    data_config = config["data"][0] if isinstance(config["data"], list) else config["data"]
    seed = int(config.get("seed", 17))

    # The same split call as `sleepwalker train`, so the extractors only ever see that run's training patients.
    dataset = build_dataset(one_epoch_entry(data_config, args.epoch))
    patients = load_patient_split(data_config["files"], dataset, seed)["train"]
    if data_config.get("patient_filter") is not None:
        patients = apply_patient_filter(data_config["patient_filter"], patients, dataset, num_workers=int(data_config.get("num_workers", 4)))
    if args.dry:
        patients = patients[:DRY_RUN_PATIENTS]
    dataset.initialize(patients, int(data_config.get("num_workers", 4)), strict=bool(data_config.get("strict", False)))

    roles = [cfg.logical_name for cfg in dataset.channels]
    featurizer = Featurizer(list(args.features), roles, float(dataset.sample_frequency), epoch_duration=args.epoch)
    featurize = Featurize(featurizer, n_jobs=args.n_jobs, fit_windows=args.fit_windows)

    # Random, seeded windows across every patient: sequential order would spend the budget on the first few.
    loader = build_loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, n_samples=None, collate_fn=batch_collate, sampling="random", seed=seed)
    warmup(featurize, loader)

    model = PackagedEmbeddingModel(
        encoder=featurize, embedding_dim=featurize.n_features, ts_len=dataset.get_timeseries_len(), n_channels=len(roles), channel_first=False
    )
    package = save_packaged_model(
        args.out,
        name="-".join(args.features),
        model=model,
        dataset=dataset,
        classification_contract=None,  # embedding-only: no head, so no contract and no task
        task=None,
        config={
            "features": list(args.features),
            "epoch": args.epoch,
            "fit_windows": int(args.fit_windows),
            "n_jobs": int(args.n_jobs),
            "seed": seed,
            "source_config": os.path.abspath(args.config),
            "n_train_patients": len(patients),
        },
    )
    logger.info(f"Packaged '{package.name}' to {args.out}: {featurize.n_features} features from {len(roles)} channels {roles}.")


if __name__ == "__main__":
    main()
