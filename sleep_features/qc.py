"""
Signal-quality (QC) checks — Layer A of unscorable-data detection.

These are cheap, physically-grounded, per-epoch checks computed directly from the
raw signal. They are model-agnostic (no training distribution required), so they
live in the feature package and can be emitted alongside features. Distributional
out-of-distribution detection (Layer B) is model-coupled and lives in model/ood.py.

An epoch flagged here is one we don't trust enough to score (dead channel,
saturation/clipping, gross artifact, or non-finite samples).
"""

from typing import Optional

import numpy as np
import pandas as pd

from .transforms import robust_std


def compute_qc_flags(
    data: pd.DataFrame,
    window: int = 10,
    samplerate: int = 1000,
    flat_eps: float = 1e-8,
    clip_frac_threshold: float = 0.05,
    amplitude_k: float = 6.0,
) -> pd.DataFrame:
    """
    Per-epoch signal-quality flags.

    Returns a DataFrame indexed by epoch with columns:
        qc_flag   : bool, True if the epoch is unscorable
        qc_reason : str, the first failing check ('' if clean)

    Checks:
        nonfinite  : NaN/inf present
        flat       : near-zero variance on EEG or EMG (dead channel)
        clipping   : a large fraction of samples pinned at the epoch min/max
        amplitude  : epoch EEG std far above the recording's robust distribution
    """
    array_len = window * samplerate
    eeg = data["eeg"].to_numpy(dtype=float)
    emg = data["emg"].to_numpy(dtype=float)
    n_epochs = (len(data) + array_len - 1) // array_len

    # Recording-level robust amplitude reference for the EEG std.
    epoch_eeg_std = np.array(
        [np.std(eeg[i * array_len : (i + 1) * array_len]) for i in range(n_epochs)]
    )
    std_med = np.median(epoch_eeg_std)
    std_scale = robust_std(epoch_eeg_std)
    amp_threshold = std_med + amplitude_k * std_scale

    reasons = []
    flags = []
    for i in range(n_epochs):
        e = eeg[i * array_len : (i + 1) * array_len]
        m = emg[i * array_len : (i + 1) * array_len]
        reason = ""

        if not (np.all(np.isfinite(e)) and np.all(np.isfinite(m))):
            reason = "nonfinite"
        elif np.std(e) < flat_eps or np.std(m) < flat_eps:
            reason = "flat"
        else:
            clip_frac = _clip_fraction(e)
            if clip_frac > clip_frac_threshold:
                reason = "clipping"
            elif np.std(e) > amp_threshold:
                reason = "amplitude"

        reasons.append(reason)
        flags.append(reason != "")

    return pd.DataFrame({"qc_flag": flags, "qc_reason": reasons})


def _clip_fraction(x: np.ndarray) -> float:
    """Fraction of samples pinned at the epoch's min or max (saturation proxy)."""
    if len(x) == 0:
        return 0.0
    hi = x.max()
    lo = x.min()
    if hi == lo:
        return 1.0
    return float(np.count_nonzero((x == hi) | (x == lo)) / len(x))
