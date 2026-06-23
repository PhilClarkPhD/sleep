"""
Backend configuration settings.
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Set

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Mora Sleep Scoring API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Security settings
    # API keys (comma-separated list). If empty, no auth required.
    API_KEYS: str = ""  # Set via MORA_API_KEYS env var

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = False  # Set via MORA_RATE_LIMIT_ENABLED
    RATE_LIMIT_PER_MINUTE: int = 60  # Requests per minute per client

    # CORS settings
    CORS_ORIGINS: str = "*"  # Comma-separated list or "*" for all

    @property
    def API_KEY_HASHES(self) -> Set[str]:
        """Get hashed API keys for comparison."""
        if not self.API_KEYS:
            return set()
        keys = [k.strip() for k in self.API_KEYS.split(",") if k.strip()]
        return {hashlib.sha256(k.encode()).hexdigest() for k in keys}

    @property
    def CORS_ORIGINS_LIST(self) -> List[str]:
        """Get CORS origins as list."""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # Model settings
    # Default: look in backend/models/ (for Docker), fallback to model_artifacts/ (for local dev)
    MODEL_PATH: Path = (
        Path(__file__).parent.parent.parent / "models" / "XGBoost_1.2.4.pkl"
        if (Path(__file__).parent.parent.parent / "models" / "XGBoost_1.2.4.pkl").exists()
        else Path(__file__).parent.parent.parent.parent / "model_artifacts" / "XGBoost_1.2.4" / "XGBoost_1.2.4.pkl"
    )

    # Feature columns expected by the model (in order)
    # Canonical list lives in model/config.py — keep in sync
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
