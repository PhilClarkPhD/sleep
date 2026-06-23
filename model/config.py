"""
Single source of truth for model constants.

Both the model training pipeline and the backend API should reference these.
"""

FEATURE_COLS = [
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

EPOCH_DURATION = 10  # seconds
DEFAULT_SAMPLERATE = 1000  # Hz

# Frequency bands for power spectrum
DELTA_BAND = (0.5, 4.0)
THETA_BAND = (5.5, 8.5)

SLEEP_STATES = ["Wake", "Non REM", "REM"]
