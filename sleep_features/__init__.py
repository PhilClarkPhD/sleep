"""
sleep_features — standalone feature-engineering package for sleep scoring.

Single source of truth for turning raw EEG/EMG signals into the feature table
consumed by both the model-training pipeline and the serving backend. This
package depends on nothing in `backend/` or `model/`; those consumers import it,
never the reverse.

Public API:
    generate_features        - raw eeg/emg DataFrame -> feature table (in-memory)
    apply_rule_based_filter  - smooth a sequence of predicted states
    compute_power, smooth_signal, compute_relative_power, compute_signal_features
                             - the underlying transforms
"""

from .filters import apply_rule_based_filter
from .normalize import normalize_features
from .pipeline import generate_feature_table, generate_features
from .reliability import ReliabilityReport, assess_recording, emg_dynamic_range
from .spectral import compute_spectral_features
from .temporal import add_temporal_context
from .transforms import (
    compute_hjorth_features,
    compute_power,
    compute_relative_power,
    compute_signal_features,
    compute_signal_features_raw,
    smooth_signal,
)

__all__ = [
    "generate_features",
    "generate_feature_table",
    "normalize_features",
    "apply_rule_based_filter",
    "compute_power",
    "smooth_signal",
    "compute_relative_power",
    "compute_signal_features",
    "compute_signal_features_raw",
    "compute_spectral_features",
    "compute_hjorth_features",
    "add_temporal_context",
    "ReliabilityReport",
    "assess_recording",
    "emg_dynamic_range",
]
