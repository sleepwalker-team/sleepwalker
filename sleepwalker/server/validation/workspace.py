import hashlib
import json
import re
import stat
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from zipfile import BadZipFile, ZipFile

from pydantic import BaseModel, ConfigDict, Field

from ..storage.temporary_storage import DEFAULT_MAX_FILE_SIZE_BYTES
from ..stores.annotation_store import AnnotationOperation
from ..stores.prediction_store import PredictionKind
from .study import StudyFormat

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveNumber = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
Text = Annotated[str, Field(strict=True, min_length=1)]


class InvalidWorkspaceError(Exception):
    pass


class UnsupportedWorkspaceVersionError(Exception):
    pass


@dataclass(frozen=True)
class WorkspaceLimits:
    max_total_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
    max_document_bytes: int = 256 * 1024**2
    max_manifest_bytes: int = 1024**2
    max_members: int = 10000


class WorkspaceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspaceStudy(WorkspaceDocument):
    duration: PositiveNumber
    source_files: list[Text] = Field(min_length=1)


class WorkspaceModel(WorkspaceDocument):
    id: Text
    task: Text
    classes: list[Text] = Field(min_length=1)
    output_kind: PredictionKind
    output_resolution: PositiveNumber
    name: Text | None = None
    version: Text | None = None
    input_channels: list[Text] | None = None
    capabilities: list[Text] | None = None


class WorkspaceManifest(WorkspaceDocument):
    format: Literal["sleepwalker"]
    format_version: Literal[1]
    created_at: datetime
    created_with: Text
    study: WorkspaceStudy
    tasks: list[Text]
    prediction_files: dict[Text, Text]
    models: list[WorkspaceModel]
    checksums: dict[Text, Text]


class WorkspaceSignal(WorkspaceDocument):
    id: Text
    label: Text
    unit: str
    sample_rate: PositiveNumber


class WorkspaceMetadata(WorkspaceDocument):
    id: Text
    created_at: datetime
    expires_at: datetime
    format: StudyFormat
    duration: PositiveNumber
    recording_start: datetime
    signals: list[WorkspaceSignal] = Field(min_length=1)


class WorkspaceTask(WorkspaceDocument):
    model_id: Text
    task: Text
    kind: PredictionKind
    classes: list[Text] = Field(min_length=1)
    resolution: PositiveNumber
    predictions: list[dict]


class WorkspaceInterval(WorkspaceDocument):
    start: Number
    end: Number
    label: Text
    probabilities: dict[Text, Number]


class WorkspaceSample(WorkspaceDocument):
    time: Number
    probabilities: dict[Text, Number]


class WorkspaceEdit(WorkspaceDocument):
    id: Text
    session_id: Text
    operation: AnnotationOperation
    created_at: datetime
    task: Text | None = None
    start: Number | None = None
    end: Number | None = None
    label: Text | None = None
    event_id: Text | None = None
    reverted_edit_id: Text | None = None


@dataclass
class LoadedWorkspace:
    manifest: WorkspaceManifest
    metadata: WorkspaceMetadata
    tasks: list[WorkspaceTask]
    edits: list[WorkspaceEdit]
    sources: list[Path]


def inspect_workspace(path: Path, limits: WorkspaceLimits) -> WorkspaceManifest:
    try:
        with ZipFile(path) as archive:
            return _inspect_archive(archive, limits)
    except (
        BadZipFile, UnicodeError, ValueError, KeyError,
        NotImplementedError, RecursionError, zlib.error,
    ) as exc:
        raise InvalidWorkspaceError("Workspace archive or manifest is invalid.") from exc


def _inspect_archive(archive: ZipFile, limits: WorkspaceLimits) -> WorkspaceManifest:
    members = archive.infolist()
    if not members or len(members) > limits.max_members:
        raise InvalidWorkspaceError("Workspace archive contains too many or no files.")

    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise InvalidWorkspaceError("Workspace archive contains duplicate filenames.")

    if sum(member.file_size for member in members) > limits.max_total_bytes:
        raise InvalidWorkspaceError("Workspace exceeds the uncompressed size limit.")

    for member in members:
        name = member.filename
        mode = member.external_attr >> 16
        if (
            not name or "\\" in name or "\x00" in member.orig_filename or ":" in name
            or PurePosixPath(name).is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or member.is_dir() or member.flag_bits & 1
            or (stat.S_IFMT(mode) not in {0, stat.S_IFREG})
        ):
            raise InvalidWorkspaceError("Workspace archive contains an unsafe file entry.")

    if "manifest.json" not in names:
        raise InvalidWorkspaceError("Workspace manifest is missing.")
    if archive.getinfo("manifest.json").file_size > limits.max_manifest_bytes:
        raise InvalidWorkspaceError("Workspace manifest exceeds the size limit.")

    raw = _read_json(archive.read("manifest.json"))
    if not isinstance(raw, dict) or raw.get("format") != "sleepwalker":
        raise InvalidWorkspaceError("Unsupported workspace format.")
    version = raw.get("format_version")
    if type(version) is not int:
        raise InvalidWorkspaceError("Workspace format version is invalid.")
    if version != 1:
        raise UnsupportedWorkspaceVersionError("Workspace format version is not supported.")

    manifest = WorkspaceManifest.model_validate(raw)
    if len(manifest.tasks) != len(set(manifest.tasks)):
        raise InvalidWorkspaceError("Workspace contains duplicate tasks.")
    if set(manifest.prediction_files) != set(manifest.tasks):
        raise InvalidWorkspaceError("Workspace prediction mapping does not match its tasks.")

    sources = manifest.study.source_files
    if len(sources) != len(set(sources)) or any(
        not name.startswith("study/source/") or len(name.split("/")) != 3
        for name in sources
    ):
        raise InvalidWorkspaceError("Workspace source file mapping is invalid.")

    predictions = list(manifest.prediction_files.values())
    if len(predictions) != len(set(predictions)) or any(
        not name.startswith("predictions/") or not name.endswith(".json")
        or len(name.split("/")) != 2 for name in predictions
    ):
        raise InvalidWorkspaceError("Workspace prediction file mapping is invalid.")

    expected = {
        "study/metadata.json", "annotations/edits.json", "provenance/models.json",
        *sources, *predictions,
    }
    if set(names) != expected | {"manifest.json"} or set(manifest.checksums) != expected:
        raise InvalidWorkspaceError("Workspace required files or checksums do not match.")
    if any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in manifest.checksums.values()):
        raise InvalidWorkspaceError("Workspace checksum is invalid.")

    for name in expected - set(sources):
        if archive.getinfo(name).file_size > limits.max_document_bytes:
            raise InvalidWorkspaceError("Workspace JSON document exceeds the size limit.")

    return manifest


def load_workspace(
    path: Path,
    source_directory: Path,
    limits: WorkspaceLimits,
    check_cancelled: Callable[[], None],
) -> LoadedWorkspace:
    try:
        with ZipFile(path) as archive:
            manifest = _inspect_archive(archive, limits)
            source_directory.mkdir(parents=True, exist_ok=False)
            sources = []
            documents = {}
            for name, expected in manifest.checksums.items():
                check_cancelled()
                checksum = hashlib.sha256()
                data = bytearray()
                source_path = source_directory / PurePosixPath(name).name
                is_source = name in manifest.study.source_files
                with archive.open(name) as entry:
                    if is_source:
                        with source_path.open("xb") as target:
                            while chunk := entry.read(1024 * 1024):
                                check_cancelled()
                                target.write(chunk)
                                checksum.update(chunk)
                    else:
                        while chunk := entry.read(1024 * 1024):
                            check_cancelled()
                            data.extend(chunk)
                            checksum.update(chunk)

                if checksum.hexdigest() != expected:
                    raise InvalidWorkspaceError("Workspace checksum verification failed.")
                if is_source:
                    sources.append(source_path)
                else:
                    documents[name] = _read_json(data)

            metadata = WorkspaceMetadata.model_validate(documents["study/metadata.json"])
            tasks = [
                WorkspaceTask.model_validate(documents[manifest.prediction_files[task]])
                for task in manifest.tasks
            ]
            raw_edits = documents["annotations/edits.json"]
            if not isinstance(raw_edits, list):
                raise InvalidWorkspaceError("Workspace annotation history must be a list.")
            edits = [WorkspaceEdit.model_validate(edit) for edit in raw_edits]
            models = documents["provenance/models.json"]
            if models != [model.model_dump(mode="json", exclude_none=True) for model in manifest.models]:
                raise InvalidWorkspaceError("Workspace model provenance does not match its manifest.")

            return LoadedWorkspace(manifest, metadata, tasks, edits, sources)
    except (
        BadZipFile, UnicodeError, ValueError, KeyError,
        NotImplementedError, RecursionError, zlib.error,
    ) as exc:
        raise InvalidWorkspaceError("Workspace archive data is invalid.") from exc


def _read_json(data: bytes | bytearray):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key.")
            result[key] = value
        return result

    return json.loads(data.decode("utf-8"), object_pairs_hook=unique_keys)
