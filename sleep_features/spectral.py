"""
Extended spectral features (v2) computed from the smoothed EEG power spectrum.

All features here are within-epoch ratios or frequency descriptors, so they are
gain-invariant and require no normalization. They target the states the current
9-feature set covers weakly — notably the sigma (spindle) band for NREM and
spectral shape descriptors for REM.
"""

from typing import Dict

import numpy as np
import pandas as pd

# Bands beyond the original delta/theta (Hz).
SIGMA_BAND = (10.0, 15.0)
BETA_BAND = (15.0, 30.0)
GAMMA_BAND = (30.0, 50.0)


def _band_rel(x: np.ndarray, axis: np.ndarray, lo: float, hi: float, total: float) -> float:
    if total <= 0:
        return 0.0
    idx = np.logical_and(axis >= lo, axis <= hi)
    return float(np.sum(x[idx]) / total)


def compute_spectral_features(
    smoothed_eeg: Dict[int, np.ndarray], freq_res: float = 0.1
) -> pd.DataFrame:
    """
    Per-epoch extended spectral features from a smoothed EEG power spectrum.

    Returns columns:
        sigma_rel, beta_rel, gamma_rel, spectral_entropy, spectral_edge_95
    """
    out = {}
    for epoch, x in smoothed_eeg.items():
        x = np.asarray(x, dtype=float)
        x = np.clip(x, 0, None)  # power is non-negative; guard tiny negatives
        axis = np.arange(len(x)) * freq_res
        total = float(np.sum(x))

        sigma = _band_rel(x, axis, *SIGMA_BAND, total)
        beta = _band_rel(x, axis, *BETA_BAND, total)
        gamma = _band_rel(x, axis, *GAMMA_BAND, total)

        # Spectral entropy: Shannon entropy of the normalized spectrum, scaled
        # to [0, 1]. High = broadband/flat (Wake), low = peaked (NREM).
        if total > 0:
            p = x / total
            nz = p[p > 0]
            entropy = float(-np.sum(nz * np.log(nz)) / np.log(len(x))) if len(x) > 1 else 0.0
            # Spectral edge: lowest frequency below which 95% of power lies.
            cumulative = np.cumsum(x) / total
            edge_idx = int(np.searchsorted(cumulative, 0.95))
            edge_idx = min(edge_idx, len(axis) - 1)
            edge = float(axis[edge_idx])
        else:
            entropy = 0.0
            edge = 0.0

        out[epoch] = [sigma, beta, gamma, entropy, edge]

    df = pd.DataFrame.from_dict(out, orient="index")
    df.columns = ["sigma_rel", "beta_rel", "gamma_rel", "spectral_entropy", "spectral_edge_95"]
    return df
