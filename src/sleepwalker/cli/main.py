"""Optional command-line access to Sleepwalker's reusable workflows."""

from __future__ import annotations

import sys

from sleepwalker.cli import evaluate, evaluate_system, foundation_models, mirror_edf_repo, split, train


COMMANDS = {
    "dry": lambda arguments: train.main(["dry", *arguments]),
    "evaluate": evaluate.main,
    "evaluate-system": evaluate_system.main,
    "foundation-model": foundation_models.main,
    "mirror-edf": mirror_edf_repo.main,
    "resume": lambda arguments: train.main(["resume", *arguments]),
    "split": split.main,
    "train": lambda arguments: train.main(["train", *arguments]),
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch one optional CLI command without changing the Python API."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print("usage: sleepwalker {" + ",".join(COMMANDS) + "} ...")
        return 0
    command = arguments.pop(0)
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command {command!r}. Choose one of: {', '.join(COMMANDS)}")
    result = COMMANDS[command](arguments)
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
