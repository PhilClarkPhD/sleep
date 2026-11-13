"""
Feature-generation pipeline: orchestrates the transforms into one feature table.

This is the in-memory entry point used by every consumer (serving, CLI, and the
batch feature-store builder). Moved verbatim from
backend/app/ml/sleep_functions.py during Phase 1.
"""

from typing import Optional

import pandas as pd

from .normalize import normalize_features
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


def generate_features(data: pd.DataFrame, start_epoch: int = 9) -> pd.DataFrame:
    """
    Run all functions to compute power spectrum and features for ML in one dataframe.

    Args:
        data: DataFrame with 'eeg' and 'emg' columns
        start_epoch: Epoch to use for normalization of other epochs

    Returns:
        DataFrame with all feature columns per epoch
    """
    eeg_power, emg_power = compute_power(data)
    smoothed_eeg, smoothed_emg = smooth_signal(eeg_power, emg_power)

    signal_features = compute_signal_features(data, start_epoch=start_epoch)
    rel_power = compute_relative_power(smoothed_eeg)

    metrics = signal_features.join(rel_power)

    return metrics


def generate_feature_table(
    data: pd.DataFrame,
    normalization: str = "robust_mad",
    window: int = 10,
    samplerate: int = 1000,
    start_epoch: Optional[int] = None,
    feature_set: str = "v1",
) -> pd.DataFrame:
    """
    Canonical feature table for the feature-store builder and serving: raw
    time-domain features + spectral relative power, with a pluggable
    per-recording normalization stage applied to the amplitude features.

    Unlike generate_features(), this path takes NO hand-picked baseline epoch in
    its default ('robust_mad') mode — normalization is automatic and gain-invariant.

    Args:
        data: DataFrame with 'eeg' and 'emg' columns
        normalization: 'robust_mad' (default), 'baseline_epoch', or 'none'
        window: Epoch window in seconds
        samplerate: Sample rate in Hz
        start_epoch: Only used by the legacy 'baseline_epoch' strategy
        feature_set: 'v1' (current 12) or 'v2' (adds sigma/beta/gamma, spectral
            entropy + edge, Hjorth params, and temporal context). The v2 path also
            applies a Hann taper before the FFT to reduce spectral leakage.

    Returns:
        DataFrame with columns matching config.FEATURE_SETS[feature_set].
    """
    extended = feature_set == "v2"
    taper = "hann" if extended else None

    eeg_power, emg_power = compute_power(
        data, window=window, samplerate=samplerate, taper=taper
    )
    smoothed_eeg, _ = smooth_signal(eeg_power, emg_power)

    raw = compute_signal_features_raw(data, window=window, samplerate=samplerate)
    rel_power = compute_relative_power(smoothed_eeg)

    table = raw.join(rel_power)

    if extended:
        spectral = compute_spectral_features(smoothed_eeg)
        hjorth = compute_hjorth_features(data, window=window, samplerate=samplerate)
        table = table.join(spectral).join(hjorth)

    table = normalize_features(table, strategy=normalization, start_epoch=start_epoch)

    if extended:
        # Temporal context is derived AFTER normalization so rolling/diff act on
        # the normalized features the model will actually see.
        table = add_temporal_context(table)

    return table
