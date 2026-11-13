"""
Tests for unscorable-data detection: Layer A signal QC (sleep_features.qc) and
Layer B distributional OOD (model.ood).
"""

import numpy as np
import pandas as pd
import pytest

from sleep_features.qc import compute_qc_flags
from model.ood import MahalanobisOOD


def _make_df(eeg, emg):
    return pd.DataFrame({"eeg": eeg.astype(float), "emg": emg.astype(float)})


def test_qc_clean_signal_has_no_flags():
    rng = np.random.default_rng(0)
    n = 10 * 1000 * 10  # 10 epochs
    df = _make_df(rng.normal(0, 100, n), rng.normal(0, 50, n))
    qc = compute_qc_flags(df)
    assert len(qc) == 10
    assert not qc["qc_flag"].any()


def test_qc_flags_flat_channel():
    rng = np.random.default_rng(1)
    sr, w = 1000, 10
    n = 3 * sr * w
    eeg = rng.normal(0, 100, n)
    emg = rng.normal(0, 50, n)
    # Make epoch 1 a dead (flat) EEG channel.
    eeg[sr * w : 2 * sr * w] = 0.0
    qc = compute_qc_flags(_make_df(eeg, emg))
    assert qc["qc_reason"].iloc[1] == "flat"
    assert bool(qc["qc_flag"].iloc[1])


def test_qc_flags_clipping():
    rng = np.random.default_rng(2)
    sr, w = 1000, 10
    n = 3 * sr * w
    eeg = rng.normal(0, 100, n)
    emg = rng.normal(0, 50, n)
    # Saturate most of epoch 2's EEG at a rail value.
    seg = slice(2 * sr * w, 3 * sr * w)
    eeg[seg] = 32767.0
    eeg[2 * sr * w : 2 * sr * w + 100] = -32768.0  # keep some spread so not 'flat'
    qc = compute_qc_flags(_make_df(eeg, emg))
    assert qc["qc_reason"].iloc[2] == "clipping"


def test_qc_flags_nonfinite():
    rng = np.random.default_rng(3)
    sr, w = 1000, 10
    n = 2 * sr * w
    eeg = rng.normal(0, 100, n)
    emg = rng.normal(0, 50, n)
    eeg[5] = np.inf
    qc = compute_qc_flags(_make_df(eeg, emg))
    assert qc["qc_reason"].iloc[0] == "nonfinite"


def test_ood_separates_inliers_from_outliers():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, size=(2000, 5))
    det = MahalanobisOOD(quantile=0.999).fit(X)

    inliers = rng.normal(0, 1, size=(500, 5))
    # ~<=1% false-positive rate at the 0.999 threshold (allow slack)
    assert det.is_ood(inliers).mean() < 0.05

    outliers = np.full((10, 5), 20.0)
    assert det.is_ood(outliers).all()


def test_ood_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, size=(500, 4))
    det = MahalanobisOOD().fit(X, feature_cols=["a", "b", "c", "d"])
    p = tmp_path / "ood.pkl"
    det.save(p)
    loaded = MahalanobisOOD.load(p)
    assert loaded.feature_cols_ == ["a", "b", "c", "d"]
    np.testing.assert_allclose(loaded.distance(X), det.distance(X))
