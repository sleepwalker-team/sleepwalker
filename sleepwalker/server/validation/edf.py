from pathlib import Path

import pyedflib

from .study import InvalidStudyError, SignalInfo, StudyFormat, StudyInfo


def validate_edf(path: Path) -> StudyInfo:
    reader = None

    try:
        reader = pyedflib.EdfReader(str(path))

        n_signals = reader.signals_in_file

        if n_signals <= 0:
            raise InvalidStudyError(
                "EDF file contains no signals."
            )

        duration = float(reader.getFileDuration())

        if duration <= 0:
            raise InvalidStudyError(
                "EDF file has no valid recording duration."
            )

        labels = [label.strip() for label in reader.getSignalLabels()]

        signals: list[SignalInfo] = []

        for index, label in enumerate(labels):
            if not label:
                raise InvalidStudyError(
                    "EDF file contains a signal without a label."
                )

            sample_rate = float(reader.getSampleFrequency(index))

            if sample_rate <= 0:
                raise InvalidStudyError(
                    f"EDF signal '{label}' has an invalid sampling frequency."
                )

            unit = reader.getPhysicalDimension(index).strip()

            signals.append(
                SignalInfo(
                    id=label,
                    label=label,
                    unit=unit,
                    sample_rate=sample_rate,
                )
            )

        return StudyInfo(
            format=StudyFormat.EDF,
            source_files=[path],
            duration=duration,
            recording_start=reader.getStartdatetime(),
            signals=signals,
        )

    except InvalidStudyError:
        raise

    except (OSError, ValueError, RuntimeError) as exc:
        raise InvalidStudyError(
            "File is not a valid or readable EDF file."
        ) from exc

    finally:
        if reader is not None:
            reader.close()
