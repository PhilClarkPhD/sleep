"""
Manifest construction for a feature-store build.

The manifest is the lineage record that makes train/serve drift detectable: it
captures exactly which code, config, normalization strategy, and inputs produced
a given feature CSV. A model artifact can then record which manifest it trained
against.
"""

from typing import List, Optional

import pandas as pd

from . import config


def build_manifest(
    csv_name: str,
    full: pd.DataFrame,
    normalization: str,
    emg_event_k: float,
    per_recording: List[dict],
    git_sha: Optional[str],
    feature_set: str = "v1",
) -> dict:
    """Assemble the manifest dict for a completed build."""
    label_counts = (
        full[config.TARGET_COL].value_counts().sort_index().to_dict()
    )
    return {
        "schema_version": 1,
        "output_csv": csv_name,
        "git_sha": git_sha,
        "feature_code_version": config.FEATURE_CODE_VERSION,
        "feature_set": feature_set,
        "config": {
            "epoch_duration": config.EPOCH_DURATION,
            "default_samplerate": config.DEFAULT_SAMPLERATE,
            "delta_band": config.DELTA_BAND,
            "theta_band": config.THETA_BAND,
            "feature_columns": config.FEATURE_SETS[feature_set],
            "model_feature_cols": config.MODEL_FEATURE_COLS,
            "feature_store_columns": config.feature_store_columns(feature_set),
        },
        "normalization": {
            "strategy": normalization,
            "emg_event_k": emg_event_k,
        },
        "totals": {
            "n_recordings": len(per_recording),
            "n_rows": int(len(full)),
            "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        },
        "recordings": per_recording,
    }
