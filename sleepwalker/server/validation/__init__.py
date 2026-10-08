from pathlib import Path

from .edf import validate_edf
from .nox import looks_like_nox_study
from .study import InvalidStudyError, StudyInfo, UnsupportedStudyFormatError


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

    # Native Nox data cannot be read by Sleepwalker yet.
    if looks_like_nox_study(files):
        raise UnsupportedStudyFormatError(
            "Native Nox studies are not supported yet. "
            "Export the study as EDF before uploading it."
        )

    raise UnsupportedStudyFormatError(
        "Uploaded files do not represent a supported study format."
    )
