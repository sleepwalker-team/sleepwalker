#!/usr/bin/env python3
"""Run a training command described by a YAML config."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any

import yaml


def load_run_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a top-level mapping in {path}.")
    if "script" not in cfg:
        raise ValueError(f"Run config must define a script: {path}")
    return cfg


def append_arg(command: list[str], key: str, value: Any) -> None:
    flag = f"--{key}"
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if value is None:
        return
    command.append(flag)
    if isinstance(value, (list, tuple)):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def build_command(cfg: dict[str, Any], python_executable: str) -> list[str]:
    command = [python_executable, str(cfg["script"])]
    args = cfg.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("Run config 'args' must be a mapping.")
    for key, value in args.items():
        append_arg(command, key, value)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("--dry-print", action="store_true", help="Print the command without running it.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used for the child command.")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    command = build_command(cfg, args.python)
    print(" ".join(command))
    if args.dry_print:
        return

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (cfg.get("env") or {}).items()})
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
