from pathlib import Path

from .edf import validate_edf
from .study import InvalidStudyError, StudyInfo, UnsupportedStudyFormatError
from .nox import looks_like_nox_study, validate_nox

def validate_study(files: list[Path]) -> StudyInfo:
    if not files:
        raise InvalidStudyError(
            "No study files were uploaded."
        )

    # EDF
    if (
        len(files) == 1
        and files[0].suffix.lower() in {".edf", ".bdf"}
    ):
        return validate_edf(files[0])

    # NOX
    if looks_like_nox_study(files):
        return validate_nox(files)

    raise UnsupportedStudyFormatError(
        "Uploaded files do not represent a supported study format."
    )