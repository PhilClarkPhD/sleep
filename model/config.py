"""
Model constants.

The canonical source of truth now lives in `sleep_features.config`. This module
re-exports the names the model pipeline already imports, so existing imports
(`from model.config import FEATURE_COLS`) keep working while there is a single
definition shared with the feature package and the backend.
"""

from sleep_features.config import (  # noqa: F401
    DEFAULT_SAMPLERATE,
    DELTA_BAND,
    EPOCH_DURATION,
    SLEEP_STATES,
    THETA_BAND,
)
from sleep_features.config import MODEL_FEATURE_COLS as FEATURE_COLS

__all__ = [
    "FEATURE_COLS",
    "EPOCH_DURATION",
    "DEFAULT_SAMPLERATE",
    "DELTA_BAND",
    "THETA_BAND",
    "SLEEP_STATES",
]
