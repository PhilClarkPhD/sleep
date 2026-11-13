"""
Tests for recording-level reliability assessment (sleep_features.reliability).

Design under test: per-epoch "Unscored" comes only from signal QC (Layer A);
OOD rate, QC-fail rate and EMG dynamic range are recording-level WARNINGS.
"""

import numpy as np
import pandas as pd

from sleep_features.reliability import (
    EMG_RANGE_WARN_RATIO,
    OOD_WARN_PERCENT,
    QC_FAIL_WARN_PERCENT,
    assess_recording,
    emg_dynamic_range,
)


def _recording(n_epochs, emg_scale_fn, sr=1000, window=10, seed=0):
    rng = np.random.default_rng(seed)
    n = n_epochs * sr * window
    eeg = rng.normal(0, 100, n)
    emg = np.concatenate(
        [rng.normal(0, emg_scale_fn(i), sr * window) for i in range(n_epochs)]
    )
    return pd.DataFrame({"eeg": eeg, "emg": emg})


def test_emg_dynamic_range_low_for_healthy_recording():
    # Half the epochs quiet (atonia), half active -> p05 << p50
    df = _recording(20, lambda i: 5.0 if i % 2 == 0 else 50.0)
    r = emg_dynamic_range(df)
    assert r < 0.3


def test_emg_dynamic_range_near_one_when_emg_flat_across_states():
    # EMG barely changes between epochs (low-gain channel, ZZ05-like)
    df = _recording(20, lambda i: 50.0)
    r = emg_dynamic_range(df)
    assert r > 0.9


def test_emg_dynamic_range_nan_on_empty():
    df = pd.DataFrame({"eeg": np.zeros(100), "emg": np.zeros(100)})
    assert np.isnan(emg_dynamic_range(df))


def test_assess_clean_recording_has_no_warnings():
    report = assess_recording(qc_flags=[False] * 1000, emg_range=0.2, ood_flags=[False] * 1000)
    assert report.level == "ok"
    assert report.warnings == []
    assert report.qc_fail_percent == 0.0
    assert report.ood_percent == 0.0


def test_assess_without_ood_detector_reports_none():
    report = assess_recording(qc_flags=[False] * 10, emg_range=0.2, ood_flags=None)
    assert report.ood_percent is None
    assert report.level == "ok"


def test_assess_flags_noisy_recording():
    qc = [True] * 3 + [False] * 97  # 3% > 1%
    report = assess_recording(qc, emg_range=0.2)
    codes = [w.code for w in report.warnings]
    assert codes == ["noisy"]
    assert report.qc_fail_percent == 3.0
    assert report.warnings[0].threshold == QC_FAIL_WARN_PERCENT


def test_assess_flags_weak_emg_contrast():
    report = assess_recording([False] * 100, emg_range=0.9)
    codes = [w.code for w in report.warnings]
    assert codes == ["weak_emg_contrast"]
    assert report.warnings[0].threshold == EMG_RANGE_WARN_RATIO


def test_assess_flags_out_of_distribution_as_recording_level_only():
    # 20% OOD epochs -> ONE recording-level warning; the per-epoch flags are
    # not turned into Unscored epochs (that is the caller's contract).
    ood = [True] * 20 + [False] * 80
    report = assess_recording([False] * 100, emg_range=0.2, ood_flags=ood)
    codes = [w.code for w in report.warnings]
    assert codes == ["out_of_distribution"]
    assert report.ood_percent == 20.0
    assert report.warnings[0].threshold == OOD_WARN_PERCENT


def test_assess_multiple_warnings_and_to_dict():
    qc = [True] * 5 + [False] * 95
    ood = [True] * 50 + [False] * 50
    report = assess_recording(qc, emg_range=0.95, ood_flags=ood)
    assert {w.code for w in report.warnings} == {"noisy", "weak_emg_contrast", "out_of_distribution"}
    d = report.to_dict()
    assert d["level"] == "warning"
    assert len(d["warnings"]) == 3
    assert isinstance(d["warnings"][0], dict)
