"""Make the adjacent ``src`` package importable for transitional tool wrappers."""

from pathlib import Path
import sys


def add_source_checkout() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    if source_root.is_dir() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
