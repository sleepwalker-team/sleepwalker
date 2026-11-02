"""Recognise uploaded study files and read their metadata."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import pyedflib


class StudyFormat(str, Enum):
    EDF = "edf"
    NOX = "nox"


class InvalidStudyError(Exception):
    pass


class UnsupportedStudyFormatError(Exception):
    pass


@dataclass
class SignalInfo:
    id: str
    label: str
    unit: str
    sample_rate: float


@dataclass
class StudyInfo:
    format: StudyFormat
    source_files: list[Path]
    duration: float | None = None
    recording_start: datetime | None = None
    signals: list[SignalInfo] | None = None


def validate_study(files: list[Path]) -> StudyInfo:
    if not files:
        raise InvalidStudyError("No study files were uploaded.")

    if len(files) == 1 and files[0].suffix.lower() in {".edf", ".bdf"}:
        return validate_edf(files[0])

    if looks_like_nox_study(files):
        raise UnsupportedStudyFormatError("Native Nox studies are not supported yet. Export the study as EDF before uploading it.")

    raise UnsupportedStudyFormatError("Uploaded files do not represent a supported study format.")


def validate_edf(path: Path) -> StudyInfo:
    try:
        reader = pyedflib.EdfReader(str(path))
    except (OSError, ValueError, RuntimeError) as exc:
        raise InvalidStudyError("File is not a valid or readable EDF file.") from exc

    try:
        if reader.signals_in_file <= 0:
            raise InvalidStudyError("EDF file contains no signals.")

        duration = float(reader.getFileDuration())
        if duration <= 0:
            raise InvalidStudyError("EDF file has no valid recording duration.")

        signals = []
        for index, label in enumerate(label.strip() for label in reader.getSignalLabels()):
            if not label:
                raise InvalidStudyError("EDF file contains a signal without a label.")

            sample_rate = float(reader.getSampleFrequency(index))
            if sample_rate <= 0:
                raise InvalidStudyError(f"EDF signal '{label}' has an invalid sampling frequency.")

            signals.append(SignalInfo(id=label, label=label, unit=reader.getPhysicalDimension(index).strip(), sample_rate=sample_rate))

        return StudyInfo(
            format=StudyFormat.EDF,
            source_files=[path],
            duration=duration,
            recording_start=reader.getStartdatetime(),
            signals=signals,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise InvalidStudyError("File is not a valid or readable EDF file.") from exc
    finally:
        reader.close()


def looks_like_nox_study(files: list[Path]) -> bool:
    suffixes = {path.suffix.lower() for path in files}
    # .nif is a raw recorder file, .ndf a recording downloaded into Noxturnal.
    return (len(files) == 1 and ".nif" in suffixes) or ".ndf" in suffixes
