"""
Model loading and management.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the XGBoost model lifecycle."""

    def __init__(self):
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.label_encoder = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, model_path: Optional[Path] = None) -> None:
        """
        Load the model from disk.

        Args:
            model_path: Path to the model .pkl file. Uses default from settings if not provided.
        """
        if model_path is None:
            model_path = settings.MODEL_PATH

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from {model_path}")

        # Model is saved as tuple: (model, metadata_dict, label_encoder)
        model, metadata, label_encoder = joblib.load(model_path)

        self.model = model
        self.metadata = metadata
        self.label_encoder = label_encoder
        self._loaded = True

        logger.info(f"Model loaded successfully: {self.version}")

    @property
    def version(self) -> str:
        """Get model version from metadata."""
        return self.metadata.get("model_version", "unknown")

    @property
    def model_name(self) -> str:
        """Get model name from metadata."""
        return self.metadata.get("model_name", "unknown")

    @property
    def classes(self) -> List[str]:
        """Get class labels."""
        if self.label_encoder is not None:
            return self.label_encoder.classes_.tolist()
        return []

    @property
    def notes(self) -> Optional[str]:
        """Get model notes from metadata."""
        return self.metadata.get("notes")

    def predict(self, features) -> List[str]:
        """
        Make predictions using the loaded model.

        Args:
            features: DataFrame or array with feature columns

        Returns:
            List of predicted class labels
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Get raw predictions (encoded integers)
        predictions = self.model.predict(features)

        # Decode to class labels
        decoded = self.label_encoder.inverse_transform(predictions)

        return decoded.tolist()


# Global model manager instance
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Dependency injection for model manager."""
    return model_manager
