from pydantic import BaseModel


class ModelResponse(BaseModel):
    id: str
    name: str
    task: str
    classes: list[str]
    input_channels: list[str]
    capabilities: list[str]
    output_resolution: float


class ModelListResponse(BaseModel):
    models: list[ModelResponse]