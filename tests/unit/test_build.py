"""
Tests for the feature-store builder: discovery/parsing, and an end-to-end build
on a small synthetic recording (no large WAVs required).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from sleep_features import config
from sleep_features.build import build_feature_store
from sleep_features.discovery import discover_recordings, parse_wav_name


def test_parse_wav_name():
    assert parse_wav_name("ZZ06_BL") == ("ZZ06", "BL")
    assert parse_wav_name("185L_BL2") == ("185L", "BL2")
    assert parse_wav_name("194LR_AD1") == ("194LR", "AD1")


def _write_synthetic_recording(directory: Path, id_day: str, n_epochs: int, sr: int = 1000):
    rng = np.random.default_rng(0)
    n = n_epochs * sr * 10
    eeg = (rng.normal(0, 100, n)).astype(np.int16)
    emg = (rng.normal(0, 50, n)).astype(np.int16)
    wavfile.write(directory / f"{id_day}.wav", sr, np.stack([eeg, emg], axis=1))
    lines = ["epoch,score"]
    for e in range(n_epochs):
        state = "Wake" if e < n_epochs // 2 else "Non REM"
        lines.append(f"{e},{state}")
    (directory / f"{id_day}_scores.txt").write_text("\n".join(lines))


def test_discovery_pairs_and_dedups(tmp_path):
    d1 = tmp_path / "folderA" / "RAT1"
    d2 = tmp_path / "folderB" / "RAT1"  # duplicate id_day in a second location
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)
    _write_synthetic_recording(d1, "RAT1_BL", n_epochs=4)
    _write_synthetic_recording(d2, "RAT1_BL", n_epochs=4)

    recs = discover_recordings(tmp_path)
    assert len(recs) == 1  # deduped by id_day
    assert recs[0].id == "RAT1" and recs[0].day == "BL"
    assert recs[0].scores_path is not None


def test_recording_without_scores_is_skipped(tmp_path):
    d = tmp_path / "RAT2"
    d.mkdir()
    rng = np.random.default_rng(1)
    n = 3 * 1000 * 10
    data = np.stack(
        [rng.normal(0, 100, n).astype(np.int16), rng.normal(0, 50, n).astype(np.int16)],
        axis=1,
    )
    wavfile.write(d / "RAT2_BL.wav", 1000, data)  # no scores file
    assert discover_recordings(tmp_path, require_scores=True) == []


def test_end_to_end_build(tmp_path):
    raw = tmp_path / "raw" / "RAT1"
    raw.mkdir(parents=True)
    _write_synthetic_recording(raw, "RAT1_BL", n_epochs=20)
    out = tmp_path / "store"

    manifest = build_feature_store(raw, out, normalization="robust_mad")

    # CSV exists, matches the column contract, one row per epoch
    csv_path = Path(manifest["_csv_path"])
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert list(df.columns) == config.FEATURE_STORE_COLUMNS
    assert len(df) == 20
    assert set(df["ID_day"].unique()) == {"RAT1_BL"}
    assert not np.isinf(df[config.GENERATED_FEATURE_COLS].to_numpy()).any()

    # Manifest captures lineage
    assert manifest["normalization"]["strategy"] == "robust_mad"
    assert manifest["totals"]["n_recordings"] == 1
    assert manifest["totals"]["n_rows"] == 20
    mpath = Path(manifest["_manifest_path"])
    assert mpath.exists()
    on_disk = json.loads(mpath.read_text())
    assert on_disk["output_csv"] == csv_path.name


# --------------------------------------------------------------- label lookup
def _touch(d: Path, *names: str):
    for n in names:
        (d / n).write_text("epoch,score\n0,Wake\n")


def test_find_scores_prefers_canonical_then_10s_then_legacy(tmp_path):
    from sleep_features.discovery import find_scores_file

    d = tmp_path / "ZZ25_BL"
    d.mkdir()
    wav = d / "ZZ25_BLSel.wav"
    wav.write_bytes(b"")

    # Only a plain web-app export: picked, but it is the untrusted fallback.
    _touch(d, "ZZ25_BLSel_scores.csv")
    assert find_scores_file(wav).name == "ZZ25_BLSel_scores.csv"

    # A hand-corrected 10 s export beats the plain export.
    _touch(d, "ZZ25_BLSel_scores (10s).csv")
    assert find_scores_file(wav).name == "ZZ25_BLSel_scores (10s).csv"

    # Canonical name beats everything.
    _touch(d, "ZZ25_BLSel_scores.txt")
    assert find_scores_file(wav).name == "ZZ25_BLSel_scores.txt"


def test_find_scores_legacy_corrected_beats_filtered(tmp_path):
    from sleep_features.discovery import find_scores_file

    d = tmp_path / "ZZ16_BL"
    d.mkdir()
    wav = d / "ZZ16_BLSel24hr.wav"
    wav.write_bytes(b"")
    _touch(d, "ZZ16_BL(filtered).txt", "ZZ16_BL.txt")
    assert find_scores_file(wav).name == "ZZ16_BL(filtered).txt"
    _touch(d, "ZZ16_BL (corrected).txt")
    assert find_scores_file(wav).name == "ZZ16_BL (corrected).txt"


def test_find_scores_10sec_variant_and_case_insensitive(tmp_path):
    from sleep_features.discovery import find_scores_file

    d = tmp_path / "ZZ27_BL"
    d.mkdir()
    wav = d / "ZZ27_BLSel.wav"
    wav.write_bytes(b"")
    _touch(d, "ZZ27_BLSel_scores.csv", "ZZ27_BLSel_10SecScores.csv")
    assert find_scores_file(wav).name == "ZZ27_BLSel_10SecScores.csv"


def test_labels_json_override_wins(tmp_path):
    from sleep_features.discovery import discover_recordings

    d = tmp_path / "ZZ30_BL"
    d.mkdir()
    _write_synthetic_recording(d, "ZZ30_BLSel", n_epochs=2)  # writes *_scores.txt
    _touch(d, "ZZ30_BLSel_scores.csv")
    # Folder-level override points at the csv even though the canonical txt exists.
    (d / "labels.json").write_text(json.dumps({"ZZ30_BLSel": "ZZ30_BLSel_scores.csv"}))
    recs = discover_recordings(tmp_path)
    assert len(recs) == 1
    assert recs[0].scores_path.name == "ZZ30_BLSel_scores.csv"

    # Root-level override also works and folder-level takes precedence.
    (tmp_path / "labels.json").write_text(json.dumps({"ZZ30_BLSel": "ZZ30_BLSel_scores.txt"}))
    assert discover_recordings(tmp_path)[0].scores_path.name == "ZZ30_BLSel_scores.csv"
    (d / "labels.json").unlink()
    assert discover_recordings(tmp_path)[0].scores_path.name == "ZZ30_BLSel_scores.txt"


def test_labels_json_null_excludes_recording(tmp_path):
    from sleep_features.discovery import discover_recordings

    good = tmp_path / "RAT1"; good.mkdir()
    bad = tmp_path / "RAT2"; bad.mkdir()
    _write_synthetic_recording(good, "RAT1_BL", n_epochs=2)
    _write_synthetic_recording(bad, "RAT2_BL", n_epochs=2)
    assert {r.id for r in discover_recordings(tmp_path)} == {"RAT1", "RAT2"}
    (tmp_path / "labels.json").write_text(json.dumps({"RAT2_BL": None}))
    assert {r.id for r in discover_recordings(tmp_path)} == {"RAT1"}
