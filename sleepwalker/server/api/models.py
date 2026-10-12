from fastapi import APIRouter, Depends, Request, status

from ..errors import APIError
from ..schemas.model import ModelListResponse, ModelResponse
from ..services.model_registry import ModelInfo, ModelRegistry

router = APIRouter(
    prefix="/api/v1/models",
    tags=["models"],
)


def get_model_registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


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


@router.get(
    "",
    response_model=ModelListResponse,
)
def list_models(
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelListResponse:
    return ModelListResponse(models=[to_model_response(model) for model in registry.list()])


@router.get(
    "/{model_id}",
    response_model=ModelResponse,
)
def get_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelResponse:
    model = registry.get(model_id)

    if model is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="MODEL_NOT_FOUND",
            message="Model not found.",
        )

    return to_model_response(model)
