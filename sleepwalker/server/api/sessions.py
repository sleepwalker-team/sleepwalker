from fastapi import APIRouter, File, UploadFile, status

from ..errors import APIError
from ..schemas.session import SessionResponse, SessionState, SignalResponse
from ..services.session_service import SessionService
from ..stores.session_store import SessionStore
from ..storage.temporary_storage import TemporaryStorage
from ..validation import InvalidStudyError, UnsupportedStudyFormatError, validate_study


router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["sessions"],
)

store = SessionStore()
service = SessionService(store)
storage = TemporaryStorage()


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    files: list[UploadFile] = File(...),
) -> SessionResponse:
    session = service.create(source_files=[])

    try:
        # Uploads temporär speichern
        saved_files = await storage.save_files(
            session.id,
            files,
        )

        # Gesamte Study erkennen und validieren
        study_info = validate_study(saved_files)
        
        session.source_files = study_info.source_files
        session.format = study_info.format
        session.duration = study_info.duration
        session.recording_start = study_info.recording_start
        session.signals = study_info.signals

    except (InvalidStudyError, UnsupportedStudyFormatError) as exc:
        storage.delete_session(session.id)
        service.delete(session.id)

        raise APIError(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="INVALID_STUDY",
            message=str(exc),
        ) from exc

    except ValueError as exc:
        # z.B. doppelter Dateiname oder ungültiger Upload
        storage.delete_session(session.id)
        service.delete(session.id)

        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_REQUEST",
            message=str(exc),
        ) from exc

    except Exception:
        # Auch bei unerwarteten Fehlern keine halbfertige Session und keine temporären Dateien liegen lassen.
        storage.delete_session(session.id)
        service.delete(session.id)
        raise

    return SessionResponse(
        id=session.id,
        state=SessionState.READY,
        created_at=session.created_at,
        expires_at=session.expires_at,
        format=session.format,
        duration=session.duration,
        recording_start=session.recording_start,
        signals=(
            [SignalResponse(**vars(signal)) for signal in session.signals]
            if session.signals is not None
            else None
        ),
    )

@router.get(
    "/{session_id}",
    response_model=SessionResponse,
)
def get_session(session_id: str):
    session = service.get(session_id)

    if session is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="SESSION_NOT_FOUND",
            message="Session not found.",
        )

    return SessionResponse(
        id=session.id,
        state=SessionState.READY,
        created_at=session.created_at,
        expires_at=session.expires_at,
        format=session.format,
        duration=session.duration,
        recording_start=session.recording_start,
        signals=(
            [SignalResponse(**vars(signal)) for signal in session.signals]
            if session.signals is not None
            else None
        ),
    )
    
    
@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_session(session_id: str):
    session = service.delete(session_id)

    if session is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="SESSION_NOT_FOUND",
            message="Session not found.",
        )

    storage.delete_session(session_id)
