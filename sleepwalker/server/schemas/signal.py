from pydantic import BaseModel


class SignalSummaryResponse(BaseModel):
    id: str
    label: str
    unit: str
    sample_rate: float
    start: float
    end: float


class SignalListResponse(BaseModel):
    signals: list[SignalSummaryResponse]


class SignalDataResponse(BaseModel):
    signal: str
    unit: str
    sample_rate: float
    start: float
    end: float
    downsampled: bool
    points: list[tuple[float, float]]
