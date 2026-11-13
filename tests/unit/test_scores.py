"""
Tests for the robust scores-file parser (both real-world formats) and the
config drift guard.
"""

from pathlib import Path

import pandas as pd
import pytest

from sleep_features import config
from sleep_features.scores import align_scores, load_scores

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SCORES = REPO_ROOT / "sample_data" / "sample_scores.txt"


def test_simple_format(tmp_path):
    p = tmp_path / "simple_scores.txt"
    p.write_text("epoch,score\n0,Wake\n1,Wake\n2,Non REM\n3,REM\n")
    s = load_scores(p)
    assert list(s.index) == [0, 1, 2, 3]
    assert list(s.values) == ["Wake", "Wake", "Non REM", "REM"]


def test_lab_format_is_reindexed_to_zero(tmp_path):
    # 1-indexed LabChart-style export with a truncated trailing line.
    p = tmp_path / "lab_scores.txt"
    p.write_text(
        "Epoch #,Start Time,End Time,Score #, Score\n"
        "1,12/14/2021 15:26:56,12/14/2021 15:27:06,2,Non REM\n"
        "2,12/14/2021 15:27:06,12/14/2021 15:27:16,2,Non REM\n"
        "3,12/14/2021 15:27:16,12/14/2021 15:27:26,1,Wake\n"
        "4\n"  # truncated trailing line -> skipped
    )
    s = load_scores(p)
    assert list(s.index) == [0, 1, 2]  # 1-indexed -> 0-indexed
    assert list(s.values) == ["Non REM", "Non REM", "Wake"]


def test_real_sample_scores_file():
    if not SAMPLE_SCORES.exists():
        pytest.skip("sample_scores.txt not present")
    s = load_scores(SAMPLE_SCORES)
    assert s.index[0] == 0
    assert set(s.unique()).issubset(set(config.SLEEP_STATES) | {config.UNSCORED_LABEL})


def test_align_fills_missing_with_unscored():
    s = pd.Series(["Wake", "REM"], index=[0, 2], name="score")
    aligned = align_scores(s, n_epochs=4)
    assert list(aligned) == ["Wake", config.UNSCORED_LABEL, "REM", config.UNSCORED_LABEL]


def test_config_matches_model_config():
    """Guard against drift until model/config.py is repointed in Phase 3."""
    import importlib.util

    model_cfg_path = REPO_ROOT / "model" / "config.py"
    spec = importlib.util.spec_from_file_location("model_config", model_cfg_path)
    model_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_cfg)

    assert config.MODEL_FEATURE_COLS == model_cfg.FEATURE_COLS
    assert config.EPOCH_DURATION == model_cfg.EPOCH_DURATION
    assert config.SLEEP_STATES == model_cfg.SLEEP_STATES
