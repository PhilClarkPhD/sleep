"""
Pluggable per-recording feature normalization.

Normalization is a separate stage from feature computation so the scaling
strategy can be swapped without touching the transforms — and so that NO
hand-picked baseline epoch (or any user input) is required by the default path.

Only amplitude-scale features are normalized. The spectral features
(delta_rel, theta_rel, theta_over_delta) are already within-epoch ratios and are
gain-invariant by construction, so they are passed through untouched. EMG_events
is a count produced against a recording-level threshold and is likewise left
as-is.

Strategies:
    "robust_mad"     (default) per-recording (x - median) / (1.4826 * MAD).
                     Automatic, robust to artifacts, gain-invariant, no user input.
    "baseline_epoch" (legacy)  divide each amplitude feature by its value at a
                     chosen epoch. Reproduces the old behavior for A/B comparison.
    "none"           pass through unchanged.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Amplitude-scale features that require per-recording normalization.
AMPLITUDE_COLS = [
    "EEG_quantile_diff",
    "EEG_quantile_80",
    "EEG_ptp",
    "EEG_ss",
    "EEG_amp",
    "EEG_std",
    "EMG_std",
    "EMG_ptp",
]


def _robust_mad(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        x = out[c].to_numpy(dtype=float)
        med = np.median(x)
        scale = 1.4826 * np.median(np.abs(x - med))
        if not np.isfinite(scale) or scale == 0:
            scale = float(np.std(x))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        out[c] = (x - med) / scale
    return out


def _baseline_epoch(
    df: pd.DataFrame, cols: Sequence[str], start_epoch: int
) -> pd.DataFrame:
    out = df.copy()
    if start_epoch < 0 or start_epoch >= len(df):
        raise ValueError(
            f"baseline start_epoch={start_epoch} out of range for {len(df)} epochs"
        )
    for c in cols:
        if c not in out.columns:
            continue
        base = float(df[c].iloc[start_epoch])
        if base == 0 or not np.isfinite(base):
            base = 1.0
        out[c] = df[c] / base
    return out


def normalize_features(
    df: pd.DataFrame,
    strategy: str = "robust_mad",
    cols: Optional[Sequence[str]] = None,
    start_epoch: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply the chosen normalization strategy to the amplitude features of a
    feature table. Returns a new DataFrame; the input is not mutated.
    """
    cols = AMPLITUDE_COLS if cols is None else cols

    if strategy == "robust_mad":
        return _robust_mad(df, cols)
    if strategy == "baseline_epoch":
        if start_epoch is None:
            raise ValueError("strategy 'baseline_epoch' requires start_epoch")
        return _baseline_epoch(df, cols, start_epoch)
    if strategy == "none":
        return df.copy()
    raise ValueError(
        f"unknown normalization strategy '{strategy}' "
        "(expected 'robust_mad', 'baseline_epoch', or 'none')"
    )
