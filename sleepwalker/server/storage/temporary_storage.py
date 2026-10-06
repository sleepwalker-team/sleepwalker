import shutil
import tempfile
from pathlib import Path

from fastapi import UploadFile


class TemporaryStorage:
    def __init__(self, root: Path | None = None):
        self.root = root or Path(tempfile.gettempdir()) / "sleepwalker"
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
                    raise ValueError(
                        f"Duplicate filename: {filename}"
                    )

                with destination.open("wb") as target:
                    while chunk := await upload.read(1024 * 1024):
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