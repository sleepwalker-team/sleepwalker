"""Copy an EDF with renamed signals, so a model finds channels its package names differently."""

import shutil
from pathlib import Path

LABEL_SIZE = 16
SIGNAL_COUNT = slice(252, 256)
FIXED_HEADER_SIZE = 256


def read_edf_labels(path: Path) -> list[str]:
    with path.open("rb") as file:
        header = file.read(FIXED_HEADER_SIZE)
        count = int(header[SIGNAL_COUNT].decode("ascii"))
        block = file.read(count * LABEL_SIZE)
    if len(block) != count * LABEL_SIZE:
        raise ValueError("EDF header is truncated.")
    return [block[i:i + LABEL_SIZE].decode("ascii").strip() for i in range(0, len(block), LABEL_SIZE)]


def relabel_edf(source: Path, target: Path, renames: dict[str, str]) -> None:
    """Write `source` to `target` with labels renamed; only header bytes change."""
    labels = read_edf_labels(source)
    unknown = set(renames) - set(labels)
    if unknown:
        raise ValueError(f"EDF has no signals named {sorted(unknown)}.")
    new_labels = [renames.get(label, label) for label in labels]
    for label in renames.values():
        if not label or len(label) > LABEL_SIZE or not label.isascii() or not label.isprintable():
            raise ValueError(f"Signal label {label!r} cannot be written to an EDF header.")
    if len({label.casefold() for label in new_labels}) != len(new_labels):
        raise ValueError("Renamed EDF signal labels must be unique.")

    shutil.copyfile(source, target)
    with target.open("r+b") as file:
        file.seek(FIXED_HEADER_SIZE)
        file.write("".join(label.ljust(LABEL_SIZE) for label in new_labels).encode("ascii"))


def channel_renames(labels: list[str], aliases: dict[str, tuple[str, ...]], mapping: dict[str, str]) -> dict[str, str]:
    """Give each mapped signal its input's first package name; rename others that would also match.

    `mapping` assigns every model input one signal label. A package can configure
    preprocessing per name, so the first name is used deliberately.
    """
    renames = {signal: aliases[channel][0] for channel, signal in mapping.items()}
    package_names = {name.casefold() for names in aliases.values() for name in names}
    taken = {label.casefold() for label in labels}
    free = (name for name in (f"unused {index}" for index in range(2 * len(labels))) if name not in taken)
    for label in labels:
        if label not in renames and label.casefold() in package_names:
            renames[label] = next(free)
    return renames
