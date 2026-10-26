import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import UploadFile


UPLOAD_CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024**3
SESSION_DIRECTORY_PATTERN = re.compile(
    r"sess_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


class FileTooLargeError(Exception):
    def __init__(self, filename: str, max_size_bytes: int) -> None:
        super().__init__(f"Uploaded file is too large: {filename}")
        self.filename = filename
        self.max_size_bytes = max_size_bytes


class TemporaryStorage:
    def __init__(self, root: Path | None = None, *, max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES) -> None:
        if max_file_size_bytes <= 0:
            raise ValueError("Maximum upload file size must be positive.")

        self.root = root or Path(tempfile.gettempdir()) / "sleepwalker"
        self.max_file_size_bytes = max_file_size_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self._owns_root = False

    @contextmanager
    def exclusive_access(self):
        import fcntl

        lock_path = self.root / ".server.lock"
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Temporary storage is already used by another server.") from exc

            self._owns_root = True
            try:
                yield
            finally:
                self._owns_root = False
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def cleanup_orphan_sessions(self, known_session_ids: set[str]) -> list[str]:
        if not self._owns_root:
            raise RuntimeError("Storage ownership is required before cleaning orphan sessions.")

        deleted = []
        for directory in self.root.iterdir():
            if (
                SESSION_DIRECTORY_PATTERN.fullmatch(directory.name) is not None
                and directory.name not in known_session_ids
                and not directory.is_symlink() and directory.is_dir()
            ):
                shutil.rmtree(directory)
                deleted.append(directory.name)

        return deleted

    async def save_files(self, session_id: str, files: list[UploadFile]) -> list[Path]:
        session_dir = self.root / session_id
        session_dir.mkdir(parents=True, exist_ok=False)

        saved_files: list[Path] = []

        try:
            for upload in files:
                if not upload.filename:
                    raise ValueError("Uploaded file has no filename.")

                # Verhindert z.B. Dateinamen wie ../../etc/passwd
                filename = Path(upload.filename).name
                destination = session_dir / filename

                if destination.exists():
                    raise ValueError(f"Duplicate filename: {filename}")

                file_size = 0
                with destination.open("wb") as target:
                    while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
                        file_size += len(chunk)

                        if file_size > self.max_file_size_bytes:
                            raise FileTooLargeError(
                                filename,
                                self.max_file_size_bytes,
                            )

                        target.write(chunk)

                saved_files.append(destination)

            return saved_files

        except Exception:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise

        finally:
            for upload in files:
                await upload.close()

    def delete_session(self, session_id: str) -> None:
        session_dir = self.root / session_id
        shutil.rmtree(session_dir, ignore_errors=True)

    def write_json_export(self, session_id: str, export_id: str, payload: dict) -> Path:
        export_directory = self.root / session_id / "exports"
        export_directory.mkdir(parents=True, exist_ok=True)
        destination = export_directory / f"{export_id}.json"
        temporary_path = export_directory / f".{export_id}.tmp"

        try:
            with temporary_path.open("w", encoding="utf-8") as target:
                json.dump(payload, target, ensure_ascii=False, allow_nan=False)

            temporary_path.replace(destination)
            return destination
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def write_workspace_export(self, session_id: str, export_id: str, documents: dict, sources: dict[str, Path], check_cancelled: Callable[[], None]) -> Path:
        export_directory = self.root / session_id / "exports"
        export_directory.mkdir(parents=True, exist_ok=True)
        destination = export_directory / f"{export_id}.sleepwalker.zip"
        temporary_path = export_directory / f".{export_id}.tmp"
        checksums = {}

        try:
            with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED) as archive:
                for name, document in documents.items():
                    if name == "manifest.json":
                        continue

                    check_cancelled()
                    data = json.dumps(document, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    archive.writestr(name, data)
                    checksums[name] = hashlib.sha256(data).hexdigest()

                for name, path in sources.items():
                    checksum = hashlib.sha256()
                    check_cancelled()
                    with path.open("rb") as source, archive.open(name, "w", force_zip64=True) as target:
                        while chunk := source.read(UPLOAD_CHUNK_SIZE):
                            check_cancelled()
                            target.write(chunk)
                            checksum.update(chunk)

                    checksums[name] = checksum.hexdigest()

                manifest = {**documents["manifest.json"], "checksums": checksums}
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                )

            check_cancelled()
            temporary_path.replace(destination)
            return destination
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def delete_export(self, session_id: str, export_id: str) -> None:
        directory = self.root / session_id / "exports"
        for suffix in (".json", ".sleepwalker.zip"):
            (directory / f"{export_id}{suffix}").unlink(missing_ok=True)
