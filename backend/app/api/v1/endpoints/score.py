"""
Scoring API endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.core.model_loader import ModelManager, get_model_manager
from app.core.security import verify_api_key
from app.schemas.scoring import (
    ErrorResponse,
    FeaturesResponse,
    ScoringResponse,
)
from app.services.scoring_service import ScoringService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/score",
    response_model=ScoringResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Score a WAV file",
    description="Upload a stereo WAV file (EEG on channel 1, EMG on channel 2) and get sleep scores.",
)
async def score_file(
    file: UploadFile = File(..., description="Stereo WAV file with EEG/EMG data"),
    start_epoch: int = Form(
        default=2,
        ge=0,
        description="Baseline epoch index for normalization (should be a Wake epoch)",
    ),
    include_signals: bool = Form(
        default=True,
        description="Include signal data (EEG/EMG/power) for visualization",
    ),
    model_manager: ModelManager = Depends(get_model_manager),
    client_id: Optional[str] = Depends(verify_api_key),
):
    """
    Score a WAV file for sleep states.

    The WAV file should be stereo with:
    - Channel 1 (left): EEG signal
    - Channel 2 (right): EMG signal

    The start_epoch parameter specifies which epoch to use as the baseline
    for feature normalization. This should be an epoch where the animal is
    in a Wake state. Default is epoch 2 (seconds 20-30 of the recording).

    Set include_signals=true (default) to include signal data for visualization.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is starting up.",
        )

    # Validate file type
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="File must be a .wav file",
        )

    # Check file size
    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    logger.info(f"Scoring file: {file.filename} ({file_size_mb:.2f}MB)")

    try:
        service = ScoringService(model_manager)
        result = service.score(
            file_bytes,
            start_epoch=start_epoch,
            include_signals=include_signals,
        )

        return ScoringResponse(
            success=True,
            epochs=result.epochs,
            summary=result.summary,
            model_version=model_manager.version,
            samplerate=result.samplerate,
            baseline_epoch=result.baseline_epoch,
            signal_data=result.signal_data,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@router.post(
    "/features",
    response_model=FeaturesResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Extract features only",
    description="Extract features from a WAV file without scoring.",
)
async def extract_features(
    file: UploadFile = File(..., description="Stereo WAV file with EEG/EMG data"),
    start_epoch: int = Form(
        default=2,
        ge=0,
        description="Baseline epoch index for normalization",
    ),
    model_manager: ModelManager = Depends(get_model_manager),
    client_id: Optional[str] = Depends(verify_api_key),
):
    """
    Extract features from a WAV file without scoring.

    Useful for debugging or manual analysis of the feature values.
    """
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="File must be a .wav file",
        )

    file_bytes = await file.read()

    try:
        service = ScoringService(model_manager)
        epochs, features_df = service.extract_features(file_bytes, start_epoch=start_epoch)

        # Convert DataFrame to dict format
        features_dict = {col: features_df[col].tolist() for col in features_df.columns}

        return FeaturesResponse(
            success=True,
            epochs=epochs,
            features=features_dict,
            feature_names=list(features_df.columns),
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")
