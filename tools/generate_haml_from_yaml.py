#!/usr/bin/env python3
"""Generate a HAMl template from a YAML run config plus sweep choices."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


SWEEP_KEY = "sweep"


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a top-level mapping in {path}.")
    return cfg


def set_path(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    current: Any = payload
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise ValueError(f"Cannot set {dotted_path}: {part} is not a mapping.")
        current = current.setdefault(part, {})
    if not isinstance(current, dict):
        raise ValueError(f"Cannot set {dotted_path}: parent is not a mapping.")
    current[parts[-1]] = value


def haml_expr(values: list[Any]) -> str:
    rendered = []
    for value in values:
        if isinstance(value, str):
            rendered.append(f'"{value}"')
        else:
            rendered.append(yaml.safe_dump(value, default_flow_style=True).strip())
    return "{{ " + " || ".join(rendered) + " }}"


def build_haml_config(cfg: dict[str, Any]) -> dict[str, Any]:
    sweep = cfg.get(SWEEP_KEY, {})
    if not isinstance(sweep, dict):
        raise ValueError("'sweep' must be a mapping from dotted path to choice list.")
    base = deepcopy({k: v for k, v in cfg.items() if k != SWEEP_KEY})
    for dotted_path, values in sweep.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Sweep entry {dotted_path!r} must be a non-empty list.")
        set_path(base, dotted_path, haml_expr(values))
    return base


def dump_haml(payload: dict[str, Any]) -> str:
    text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, width=10_000)
    return text.replace("'", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="YAML run config containing an optional 'sweep' mapping.")
    parser.add_argument("--output", type=str, default=None, help="Output .hml path. Defaults next to input.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    payload = build_haml_config(cfg)
    output = Path(args.output) if args.output else Path(args.config).with_suffix(".hml")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dump_haml(payload), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
