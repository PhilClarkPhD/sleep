"""
Model information API endpoints.
"""

from typing import Optional

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.model_loader import ModelManager, get_model_manager
from app.core.security import verify_api_key
from app.schemas.scoring import HealthResponse, ModelInfoResponse

router = APIRouter()


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get model information",
    description="Returns metadata about the currently loaded model.",
)
async def get_model_info(
    model_manager: ModelManager = Depends(get_model_manager),
    client_id: Optional[str] = Depends(verify_api_key),
):
    """
    Get information about the loaded model.

    Returns model name, version, expected features, and class labels.
    """
    return ModelInfoResponse(
        model_name=model_manager.model_name,
        model_version=model_manager.version,
        feature_columns=settings.FEATURE_COLS,
        classes=model_manager.classes,
        notes=model_manager.notes,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API is running and the model is loaded.",
)
async def health_check(
    model_manager: ModelManager = Depends(get_model_manager),
):
    """
    Health check endpoint.

    Returns the API status, whether the model is loaded, and whether auth is required.
    This endpoint is always public (no auth) for infrastructure health checks.
    """
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        version=settings.VERSION,
        auth_required=bool(settings.API_KEYS),
    )
