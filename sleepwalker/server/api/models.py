from fastapi import APIRouter

from ..inference.catalog import ModelInfo
from .deps import Catalog
from .errors import not_found
from .schemas import ModelListResponse, ModelResponse

router = APIRouter(prefix="/api/v1/models", tags=["models"])


def to_model_response(model: ModelInfo) -> ModelResponse:
    return ModelResponse(
        id=model.id,
        name=model.name,
        task=model.task,
        classes=list(model.classes),
        input_channels=list(model.input_channels),
        capabilities=list(model.capabilities),
        output_kind=model.output_kind,
        output_resolution=model.output_resolution,
    )


@router.get("", response_model=ModelListResponse)
def list_models(catalog: Catalog) -> ModelListResponse:
    return ModelListResponse(models=[to_model_response(model) for model in catalog.list()])


@router.get("/{model_id}", response_model=ModelResponse)
def get_model(model_id: str, catalog: Catalog) -> ModelResponse:
    model = catalog.get(model_id)
    if model is None:
        raise not_found("MODEL_NOT_FOUND", "Model not found.")
    return to_model_response(model)
