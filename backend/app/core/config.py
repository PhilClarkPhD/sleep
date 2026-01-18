"""
Backend configuration settings.
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Mora Sleep Scoring API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Model settings
    # Default: look in backend/models/ (for Docker), fallback to model_artifacts/ (for local dev)
    MODEL_PATH: Path = (
        Path(__file__).parent.parent.parent / "models" / "XGBoost_1.2.4.pkl"
        if (Path(__file__).parent.parent.parent / "models" / "XGBoost_1.2.4.pkl").exists()
        else Path(__file__).parent.parent.parent.parent / "model_artifacts" / "XGBoost_1.2.4" / "XGBoost_1.2.4.pkl"
    )

    # Feature columns expected by the model (in order)
    FEATURE_COLS: List[str] = [
        "EEG_quantile_80",
        "EEG_ptp",
        "EEG_ss",
        "EMG_std",
        "EMG_events",
        "EMG_ptp",
        "delta_rel",
        "theta_rel",
        "theta_over_delta",
    ]

    # File upload settings
    MAX_UPLOAD_SIZE_MB: int = 500
    ASYNC_THRESHOLD_MB: int = 50

    # Processing settings
    EPOCH_DURATION_SECONDS: int = 10
    DEFAULT_SAMPLERATE: int = 1000

    class Config:
        env_prefix = "MORA_"
        case_sensitive = True


settings = Settings()
