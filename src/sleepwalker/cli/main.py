"""Optional command-line access to Sleepwalker's reusable workflows."""

from __future__ import annotations

import argparse
import sys

from sleepwalker.cli import evaluate, evaluate_system, foundation_models, mirror_edf_repo, split, train
from sleepwalker.utils import logger


COMMANDS = {
    "dry": lambda arguments, run_id=None: train.main(["dry", *arguments], run_id=run_id),
    "evaluate": evaluate.main,
    "evaluate-system": evaluate_system.main,
    "foundation-model": foundation_models.main,
    "mirror-edf": mirror_edf_repo.main,
    "resume": lambda arguments, run_id=None: train.main(["resume", *arguments], run_id=run_id),
    "split": split.main,
    "train": lambda arguments, run_id=None: train.main(["train", *arguments], run_id=run_id),
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch one optional CLI command without changing the Python API."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="sleepwalker", add_help=False, allow_abbrev=False)
    parser.add_argument("--id", dest="run_id", help="External run identifier, for example supplied by HAML.")
    options, arguments = parser.parse_known_args(arguments)
    if options.run_id is not None and not options.run_id.strip():
        parser.error("--id must not be empty.")
    if not arguments or arguments[0] in {"-h", "--help"}:
        print("usage: sleepwalker [--id RUN_ID] {" + ",".join(COMMANDS) + "} ...")
        print("--id RUN_ID: external run identifier; accepted before or after the command.")
        return 0
    command = arguments.pop(0)
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command {command!r}. Choose one of: {', '.join(COMMANDS)}")
    if options.run_id is not None:
        logger.context(options.run_id)
    try:
        if command in {"train", "dry", "resume"}:
            result = COMMANDS[command](arguments, run_id=options.run_id)
        else:
            result = COMMANDS[command](arguments)
        return 0 if result is None else int(result)
    finally:
        if options.run_id is not None:
            logger.uncontext()


if __name__ == "__main__":
    raise SystemExit(main())
