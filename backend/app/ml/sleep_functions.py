"""
DEPRECATED SHIM — feature code now lives in the standalone `sleep_features` package.

This module used to contain the feature-engineering math. It was extracted into
the repo-root `sleep_features/` package so that the model pipeline, CLI, and
backend can all share one source of truth without depending on the backend.

This file re-exports the public API so existing imports
(`from app.ml import sleep_functions as sleep`) keep working unchanged. New code
should import from `sleep_features` directly. It will be removed once all
consumers are cut over (Phase 3).
"""

from sleep_features import (  # noqa: F401
    apply_rule_based_filter,
    compute_power,
    compute_relative_power,
    compute_signal_features,
    generate_features,
    smooth_signal,
)

__all__ = [
    "generate_features",
    "apply_rule_based_filter",
    "compute_power",
    "smooth_signal",
    "compute_relative_power",
    "compute_signal_features",
]
