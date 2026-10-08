from pathlib import Path


def looks_like_nox_study(files: list[Path]) -> bool:
    if not files:
        return False

    suffixes = {
        path.suffix.lower()
        for path in files
    }

    # Raw recording from a Nox recorder
    if len(files) == 1 and ".nif" in suffixes:
        return True

    # Recording already downloaded into Noxturnal
    if ".ndf" in suffixes:
        return True

    return False
