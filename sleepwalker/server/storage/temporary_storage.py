import shutil
import tempfile
from pathlib import Path

from fastapi import UploadFile


UPLOAD_CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024**3


class FileTooLargeError(Exception):
    def __init__(self, filename: str, max_size_bytes: int) -> None:
        super().__init__(f"Uploaded file is too large: {filename}")
        self.filename = filename
        self.max_size_bytes = max_size_bytes


class TemporaryStorage:
    def __init__(
        self,
        root: Path | None = None,
        *,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    ) -> None:
        if max_file_size_bytes <= 0:
            raise ValueError("Maximum upload file size must be positive.")

        self.root = root or Path(tempfile.gettempdir()) / "sleepwalker"
        self.max_file_size_bytes = max_file_size_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    async def save_files(
        self,
        session_id: str,
        files: list[UploadFile],
    ) -> list[Path]:
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
