"""
Tests for the extended (v2) feature set: column contract, finiteness,
gain-invariance (still holds for the new spectral/Hjorth features), and the
temporal-context derivation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

import sleep_features as sf
from sleep_features import config
from sleep_features.temporal import add_temporal_context

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_WAV = REPO_ROOT / "sample_data" / "sample_signal.wav"


@pytest.fixture(scope="module")
def sample_df():
    if not SAMPLE_WAV.exists():
        pytest.skip("sample WAV not present")
    sr, data = wavfile.read(SAMPLE_WAV)
    n = 120 * sr * 10
    return pd.DataFrame(
        {"eeg": data[:n, 0].astype(np.float64), "emg": data[:n, 1].astype(np.float64)}
    )


def test_v2_columns_match_contract(sample_df):
    table = sf.generate_feature_table(sample_df, feature_set="v2")
    assert list(table.columns) == config.FEATURE_SETS["v2"]
    # v2 is a superset of v1
    assert set(config.FEATURE_SETS["v1"]).issubset(set(table.columns))


def test_v2_no_nan_or_inf(sample_df):
    table = sf.generate_feature_table(sample_df, feature_set="v2")
    arr = table.to_numpy(dtype=float)
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()


def test_v2_gain_invariance(sample_df):
    """The added spectral/Hjorth features are gain-invariant too.

    Temporal columns are derived from gain-invariant features, so the whole v2
    table is invariant to a global signal scaling.
    """
    base = sf.generate_feature_table(sample_df, feature_set="v2")
    scaled = sf.generate_feature_table(sample_df * 3.0, feature_set="v2")
    np.testing.assert_allclose(
        base.to_numpy(dtype=float), scaled.to_numpy(dtype=float), rtol=1e-6, atol=1e-6
    )


def test_spectral_relative_powers_in_range(sample_df):
    table = sf.generate_feature_table(sample_df, feature_set="v2")
    for c in ["sigma_rel", "beta_rel", "gamma_rel", "spectral_entropy"]:
        assert table[c].between(-1e-9, 1.0 + 1e-9).all()


def test_temporal_context_values():
    df = pd.DataFrame({"delta_rel": [0.0, 1.0, 2.0, 3.0], "EMG_std": [0.0, 0.0, 0.0, 0.0]})
    out = add_temporal_context(df, cols=["delta_rel"], window=3)
    # centered rolling mean with min_periods=1
    assert out["delta_rel_roll3"].iloc[0] == pytest.approx(0.5)  # mean(0,1)
    assert out["delta_rel_roll3"].iloc[1] == pytest.approx(1.0)  # mean(0,1,2)
    # first difference, first row filled with 0
    assert out["delta_rel_diff1"].iloc[0] == 0.0
    assert out["delta_rel_diff1"].iloc[2] == pytest.approx(1.0)


def test_v1_still_default_and_unchanged_contract(sample_df):
    table = sf.generate_feature_table(sample_df)  # default v1
    assert list(table.columns) == config.FEATURE_SETS["v1"]
