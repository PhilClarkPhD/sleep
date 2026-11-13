"""
OOD detection for the model pipeline.

The detector class lives in ``sleep_features.ood`` so the serving backend can
deserialize it without importing the training package. This module re-exports it
for the training pipeline's ergonomics.
"""

from sleep_features.ood import MahalanobisOOD  # noqa: F401

__all__ = ["MahalanobisOOD"]
