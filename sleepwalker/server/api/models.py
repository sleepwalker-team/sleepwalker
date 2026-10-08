from fastapi import APIRouter, status

from ..schemas.model import ModelListResponse, ModelResponse
from ..services.model_registry import ModelInfo, ModelRegistry
from ..errors import APIError

router = APIRouter(
    prefix="/api/v1/models",
    tags=["models"],
)

registry = ModelRegistry()


def to_model_response(model: ModelInfo) -> ModelResponse:
    return ModelResponse( #beschreibt verfügbares Modell, also noch keine Vorhersage
        id=model.id,
        name=model.name,
        task=model.task,
        classes=list(model.classes),
        input_channels=list(model.input_channels),
        capabilities=list(model.capabilities),
        output_resolution=model.output_resolution,
    )


@router.get(
    "",
    response_model=ModelListResponse,
)
def list_models() -> ModelListResponse:
    return ModelListResponse(
        models=[
            to_model_response(model)
            for model in registry.list()
        ]
    )
    
@router.get(
    "/{model_id}",
    response_model=ModelResponse,
)
def get_model(model_id: str) -> ModelResponse:
    model = registry.get(model_id)

    if model is None:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="MODEL_NOT_FOUND",
            message="Model not found.",
        )

    return to_model_response(model)