from pydantic import BaseModel

from ..stores.session_store import SessionState


class ImportAcceptedResponse(BaseModel):
    session_id: str
    job_id: str
    state: SessionState
