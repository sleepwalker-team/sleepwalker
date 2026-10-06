from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

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