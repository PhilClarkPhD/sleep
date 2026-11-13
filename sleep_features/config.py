"""
Canonical constants for sleep-scoring feature engineering.

This is the single source of truth for feature names, epoch geometry, and
frequency bands. Both the model pipeline and the serving backend should
reference these (the cutover from model/config.py happens in Phase 3).
"""

# Feature-code version (semver) — the compatibility contract between a trained
# model and the feature code that serves it. Bump:
#   MAJOR  when feature MATH changes (existing column values would differ)
#   MINOR  for additive, backward-compatible changes (new columns, old unchanged)
#   PATCH  for non-behavioral changes (docs, refactors that preserve outputs)
# A model records the version it trained against; serving compares (see
# check_feature_compatibility): different MAJOR -> error, different MINOR/PATCH
# -> warning.
FEATURE_CODE_VERSION = "1.0.0"


def check_feature_compatibility(model_version, serving_version=FEATURE_CODE_VERSION):
    """
    Compare a model's training-time feature-code version against the serving
    version. Returns (level, message) where level is 'ok', 'warn', or 'error'.

    - missing model version            -> warn (legacy model, can't verify)
    - different MAJOR                   -> error (feature math may differ)
    - same MAJOR, different MINOR/PATCH -> warn
    - identical                        -> ok
    """
    if not model_version:
        return ("warn", "Model has no feature_code_version (legacy); cannot verify "
                f"feature compatibility against serving v{serving_version}.")
    if model_version == serving_version:
        return ("ok", "")
    model_major = str(model_version).split(".")[0]
    serving_major = str(serving_version).split(".")[0]
    if model_major != serving_major:
        return ("error", f"Incompatible feature code: model trained on "
                f"v{model_version}, serving v{serving_version} (MAJOR differs; "
                "feature math may have changed). Refusing to serve.")
    return ("warn", f"Feature-code version mismatch: model trained on "
            f"v{model_version}, serving v{serving_version} (compatible MAJOR).")


# Epoch geometry
EPOCH_DURATION = 10  # seconds
DEFAULT_SAMPLERATE = 1000  # Hz

# Frequency bands (Hz) for spectral power
DELTA_BAND = (0.5, 4.0)
THETA_BAND = (5.5, 8.5)

# Sleep state labels
SLEEP_STATES = ["Wake", "Non REM", "REM"]
UNSCORED_LABEL = "Unscored"

# All feature columns produced by generate_features() (the full table).
GENERATED_FEATURE_COLS = [
    "EEG_quantile_diff",
    "EEG_quantile_80",
    "EEG_ptp",
    "EEG_ss",
    "EEG_amp",
    "EEG_std",
    "EMG_std",
    "EMG_events",
    "EMG_ptp",
    "delta_rel",
    "theta_rel",
    "theta_over_delta",
]

# The subset actually consumed by the current model.
MODEL_FEATURE_COLS = [
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

# --- Extended (v2) feature set -------------------------------------------------
# Additional gain-invariant spectral descriptors (sleep_features.spectral).
EXTENDED_SPECTRAL_COLS = [
    "sigma_rel",
    "beta_rel",
    "gamma_rel",
    "spectral_entropy",
    "spectral_edge_95",
]
# Scale-invariant time-domain complexity descriptors (transforms.compute_hjorth_features).
HJORTH_COLS = [
    "EEG_hjorth_mobility",
    "EEG_hjorth_complexity",
    "EMG_hjorth_mobility",
]
# Temporal-context columns are derived from these core features
# (see sleep_features.temporal); names are generated as <col>_roll3 / <col>_diff1.
TEMPORAL_CONTEXT_SOURCE_COLS = ["delta_rel", "theta_over_delta", "EMG_std", "sigma_rel"]


def temporal_context_cols():
    cols = []
    for c in TEMPORAL_CONTEXT_SOURCE_COLS:
        cols.append(f"{c}_roll3")
        cols.append(f"{c}_diff1")
    return cols


# All columns produced by the extended (v2) pipeline.
GENERATED_FEATURE_COLS_V2 = (
    GENERATED_FEATURE_COLS
    + EXTENDED_SPECTRAL_COLS
    + HJORTH_COLS
    + temporal_context_cols()
)

# Named feature sets available for model experiments.
FEATURE_SETS = {
    "v1": GENERATED_FEATURE_COLS,
    "v2": GENERATED_FEATURE_COLS_V2,
}

# Identifier/metadata columns attached by the feature-store builder.
ID_COLS = ["ID", "day", "ID_day", "epoch"]
TARGET_COL = "score"


def feature_store_columns(feature_set: str = "v1"):
    """Full feature-store table contract (column order) for a feature set."""
    return ID_COLS + FEATURE_SETS[feature_set] + [TARGET_COL]


# Default (v1) feature-store table contract.
FEATURE_STORE_COLUMNS = feature_store_columns("v1")
