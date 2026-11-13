"""
Tests for the decoupled, pluggable normalization path (generate_feature_table).

The headline guarantee: with automatic per-recording normalization there is NO
baseline epoch and NO user input, and the resulting feature table is invariant
to a global gain change on the raw signal — which is exactly the cross-recording
/ cross-animal comparability the hand-picked baseline epoch was trying (poorly)
to provide.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

import sleep_features as sf
from sleep_features import config
from sleep_features.normalize import AMPLITUDE_COLS, normalize_features

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_WAV = REPO_ROOT / "sample_data" / "sample_signal.wav"


@pytest.fixture(scope="module")
def sample_df():
    if not SAMPLE_WAV.exists():
        pytest.skip("sample WAV not present")
    sr, data = wavfile.read(SAMPLE_WAV)
    # First 120 epochs is plenty and keeps the test fast.
    n = 120 * sr * 10
    return pd.DataFrame(
        {
            "eeg": data[:n, 0].astype(np.float64),
            "emg": data[:n, 1].astype(np.float64),
        }
    )


def test_columns_match_contract(sample_df):
    table = sf.generate_feature_table(sample_df, normalization="robust_mad")
    assert set(table.columns) == set(config.GENERATED_FEATURE_COLS)


def test_no_nan_or_inf(sample_df):
    table = sf.generate_feature_table(sample_df, normalization="robust_mad")
    arr = table.to_numpy(dtype=float)
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()


def test_gain_invariance(sample_df):
    """Scaling the raw signal by a constant must not change the feature table."""
    scaled = sample_df * 7.5
    base = sf.generate_feature_table(sample_df, normalization="robust_mad")
    scaled_tbl = sf.generate_feature_table(scaled, normalization="robust_mad")

    np.testing.assert_allclose(
        base.to_numpy(dtype=float),
        scaled_tbl.to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )


def test_robust_mad_centers_amplitude_features(sample_df):
    """Robust-normalized amplitude features have ~zero median by construction."""
    table = sf.generate_feature_table(sample_df, normalization="robust_mad")
    for c in AMPLITUDE_COLS:
        assert abs(float(np.median(table[c]))) < 1e-9


def test_spectral_columns_untouched_by_normalization():
    """delta/theta features must pass through normalization unchanged."""
    df = pd.DataFrame(
        {
            "EEG_ptp": [1.0, 2.0, 3.0, 4.0],
            "delta_rel": [0.1, 0.2, 0.3, 0.4],
            "theta_over_delta": [1.5, 2.5, 0.5, 1.0],
        }
    )
    out = normalize_features(df, strategy="robust_mad", cols=["EEG_ptp"])
    pd.testing.assert_series_equal(out["delta_rel"], df["delta_rel"])
    pd.testing.assert_series_equal(out["theta_over_delta"], df["theta_over_delta"])


def test_baseline_epoch_strategy_normalizes_to_one(sample_df):
    """Legacy strategy: amplitude features equal 1.0 at the chosen baseline epoch."""
    table = sf.generate_feature_table(
        sample_df, normalization="baseline_epoch", start_epoch=9
    )
    for c in AMPLITUDE_COLS:
        assert table[c].iloc[9] == pytest.approx(1.0)


def test_unknown_strategy_raises():
    df = pd.DataFrame({"EEG_ptp": [1.0, 2.0]})
    with pytest.raises(ValueError, match="unknown normalization strategy"):
        normalize_features(df, strategy="bogus")
