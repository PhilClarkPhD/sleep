"""
Pydantic schemas for scoring API requests and responses.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ScoringRequest(BaseModel):
    """Request parameters for scoring (sent with file upload)."""

    start_epoch: int = Field(
        default=2,
        ge=0,
        description="Baseline epoch index for feature normalization (should be a Wake epoch)"
    )


class EpochScore(BaseModel):
    """A single epoch's score."""

    epoch: int = Field(description="Epoch index (0-based)")
    score: str = Field(description="Sleep state: 'Wake', 'Non REM', or 'REM'")
    timestamp_seconds: float = Field(description="Start time of epoch in seconds")


class SignalData(BaseModel):
    """Signal data for visualization (downsampled for efficiency)."""

    eeg: List[List[float]] = Field(description="EEG signal per epoch (downsampled)")
    emg: List[List[float]] = Field(description="EMG signal per epoch (downsampled)")
    power_spectrum: List[List[float]] = Field(description="Power spectrum per epoch (0-50Hz)")
    time_axis: List[float] = Field(description="Time points within epoch (seconds)")
    freq_axis: List[float] = Field(description="Frequency axis for power spectrum (Hz)")
    delta_power: List[float] = Field(description="Relative delta power per epoch")
    theta_power: List[float] = Field(description="Relative theta power per epoch")


class ScoringStats(BaseModel):
    """Summary statistics for a scoring session."""

    total_epochs: int
    wake_count: int
    wake_percent: float
    nrem_count: int
    nrem_percent: float
    rem_count: int
    rem_percent: float
    recording_duration_seconds: float
    recording_duration_hours: float


class ScoringResponse(BaseModel):
    """Response from the scoring endpoint."""

    success: bool
    epochs: List[EpochScore]
    unfiltered_epochs: List[EpochScore]
    summary: ScoringStats
    model_version: str
    samplerate: int
    baseline_epoch: int
    signal_data: Optional[SignalData] = Field(
        default=None,
        description="Signal data for visualization (included when include_signals=true)"
    )


class FeaturesResponse(BaseModel):
    """Response from the features-only endpoint."""

    success: bool
    epochs: List[int]
    features: Dict[str, List[float]]  # feature_name -> list of values per epoch
    feature_names: List[str]


class ModelInfoResponse(BaseModel):
    """Response from the model info endpoint."""

    model_name: str
    model_version: str
    feature_columns: List[str]
    classes: List[str]
    notes: Optional[str] = None


class HealthResponse(BaseModel):
    """Response from the health check endpoint."""

    status: str
    model_loaded: bool
    version: str
    auth_required: bool = Field(
        default=False,
        description="Whether API key authentication is required for other endpoints"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""

    success: bool = False
    error: str
    detail: Optional[str] = None
