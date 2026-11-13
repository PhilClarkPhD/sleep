"""
Characterization tests for the sleep_features package.

These lock in the CURRENT numeric behavior of feature generation so the Phase 1
extraction (and any later refactor) can be proven byte-for-byte equivalent.
The golden snapshot in tests/golden/generate_features_sample.csv was produced by
running the pre-extraction code on sample_data/sample_signal.wav.

If a future change is *intended* to alter feature values, regenerate the golden
file deliberately and review the diff — do not edit it casually.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

import sleep_features as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_WAV = REPO_ROOT / "sample_data" / "sample_signal.wav"
GOLDEN = REPO_ROOT / "tests" / "golden" / "generate_features_sample.csv"


@pytest.fixture(scope="module")
def sample_signal_df():
    if not SAMPLE_WAV.exists():
        pytest.skip(f"sample WAV not found: {SAMPLE_WAV}")
    samplerate, data = wavfile.read(SAMPLE_WAV)
    assert samplerate == 1000
    return pd.DataFrame(
        {"eeg": data[:, 0].astype(np.float64), "emg": data[:, 1].astype(np.float64)}
    )


def test_generate_features_matches_golden(sample_signal_df):
    """The extracted package must reproduce the pre-extraction feature table."""
    if not GOLDEN.exists():
        pytest.skip(f"golden snapshot not found: {GOLDEN}")

    expected = pd.read_csv(GOLDEN, index_col=0)
    actual = sf.generate_features(sample_signal_df)

    assert list(actual.columns) == list(expected.columns)
    assert len(actual) == len(expected)
    np.testing.assert_allclose(
        actual.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )


def test_generate_features_no_nan_or_inf(sample_signal_df):
    """Invariant: feature generation on clean input yields finite values."""
    feats = sf.generate_features(sample_signal_df)
    arr = feats.to_numpy(dtype=float)
    assert not np.isnan(arr).any(), "feature table contains NaN"
    assert not np.isinf(arr).any(), "feature table contains inf"


def test_backend_shim_reexports_same_objects():
    """The deprecated backend shim must point at the package, not a copy."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from app.ml import sleep_functions as shim

    assert shim.generate_features is sf.generate_features
    assert shim.apply_rule_based_filter is sf.apply_rule_based_filter


def test_rule_based_filter_smooths_isolated_epoch():
    """Sanity check on the rule-based filter's core behavior."""
    seq = ["Wake", "Wake", "REM", "Wake", "Wake"]
    out = sf.apply_rule_based_filter(seq)
    assert out[2] == "Wake"
    # input is not mutated
    assert seq[2] == "REM"
