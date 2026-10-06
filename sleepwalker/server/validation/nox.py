from pathlib import Path

from .study import InvalidStudyError, StudyFormat, StudyInfo


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


def validate_nox(files: list[Path]) -> StudyInfo:
    if not files:
        raise InvalidStudyError(
            "No Nox study files were uploaded."
        )

    nif_files = [
        path
        for path in files
        if path.suffix.lower() == ".nif"
    ]

    ndf_files = [
        path
        for path in files
        if path.suffix.lower() == ".ndf"
    ]

    if nif_files:
        if len(nif_files) != 1 or len(files) != 1:
            raise InvalidStudyError(
                "A raw Nox recording must contain exactly one NIF file."
            )

        nif = nif_files[0]

        _validate_file(nif)

        return StudyInfo(
            format=StudyFormat.NOX,
            source_files=[nif],
        )

    if ndf_files:
        for path in files:
            _validate_file(path)

        return StudyInfo(
            format=StudyFormat.NOX,
            source_files=files,
        )

    raise InvalidStudyError(
        "Uploaded files do not represent a recognized Nox recording."
    )


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise InvalidStudyError(
            f"Uploaded file does not exist: {path.name}"
        )

    if not path.is_file():
        raise InvalidStudyError(
            f"Uploaded path is not a file: {path.name}"
        )

    if path.stat().st_size == 0:
        raise InvalidStudyError(
            f"Uploaded file is empty: {path.name}"
        )