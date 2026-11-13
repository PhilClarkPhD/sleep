"""
Feature-store builder — the operator that turns a directory of raw recordings
into a single versioned, labeled feature CSV plus a manifest describing exactly
how it was produced.

This is the committed, reproducible replacement for the old ad-hoc notebooks.
"""

import datetime
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import config
from .build_meta import build_manifest
from .discovery import Recording, discover_recordings
from .io import load_wav
from .pipeline import generate_feature_table
from .scores import align_scores, load_scores

log = logging.getLogger(__name__)


def _read_baseline_epoch(rec: Recording) -> Optional[int]:
    if rec.epoch_path is None:
        return None
    try:
        return int(rec.epoch_path.read_text().strip())
    except (ValueError, OSError):
        log.warning("Could not read baseline epoch for %s", rec.id_day)
        return None


def build_recording_table(
    rec: Recording,
    normalization: str = "robust_mad",
    emg_event_k: float = 2.0,
    feature_set: str = "v1",
) -> pd.DataFrame:
    """Featurize one recording and attach id/epoch/score columns."""
    df, samplerate = load_wav(rec.wav_path)

    start_epoch = None
    if normalization == "baseline_epoch":
        start_epoch = _read_baseline_epoch(rec)
        if start_epoch is None:
            raise ValueError(
                f"{rec.id_day}: 'baseline_epoch' normalization needs a *_epoch.txt"
            )

    table = generate_feature_table(
        df,
        normalization=normalization,
        samplerate=samplerate,
        start_epoch=start_epoch,
        feature_set=feature_set,
    )
    n = len(table)

    scores = align_scores(load_scores(rec.scores_path), n)

    table.insert(0, "epoch", range(n))
    table.insert(0, "ID_day", rec.id_day)
    table.insert(0, "day", rec.day)
    table.insert(0, "ID", rec.id)
    table[config.TARGET_COL] = scores.values

    return table[config.feature_store_columns(feature_set)]


def build_feature_store(
    input_dir,
    out_dir,
    normalization: str = "robust_mad",
    emg_event_k: float = 2.0,
    feature_set: str = "v1",
    limit: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> dict:
    """
    Build a versioned feature store from raw recordings.

    Args:
        input_dir: directory of raw recordings (walked recursively)
        out_dir: where to write features_<ts>.csv and its manifest
        normalization: 'robust_mad' (default), 'baseline_epoch', or 'none'
        emg_event_k: EMG event-threshold multiplier
        limit: only build the first N recordings (for quick iteration)
        timestamp: override the version stamp (defaults to now)

    Returns:
        The manifest dict (also written to disk).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recordings = discover_recordings(input_dir, require_scores=True)
    if limit is not None:
        recordings = recordings[:limit]
    if not recordings:
        raise RuntimeError(f"No labeled recordings found under {input_dir}")

    log.info("Building feature store from %d recordings", len(recordings))

    frames: List[pd.DataFrame] = []
    per_recording = []
    for i, rec in enumerate(recordings, 1):
        log.info("[%d/%d] %s", i, len(recordings), rec.id_day)
        table = build_recording_table(rec, normalization, emg_event_k, feature_set)
        frames.append(table)
        n_scored = int((table[config.TARGET_COL] != config.UNSCORED_LABEL).sum())
        per_recording.append(
            {
                "id_day": rec.id_day,
                "id": rec.id,
                "day": rec.day,
                "wav": str(rec.wav_path),
                "scores": str(rec.scores_path),
                "n_epochs": int(len(table)),
                "n_scored": n_scored,
            }
        )

    full = pd.concat(frames, ignore_index=True)

    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_name = f"features_{timestamp}.csv"
    csv_path = out_dir / csv_name
    full.to_csv(csv_path, index=False)
    log.info("Wrote %d rows -> %s", len(full), csv_path)

    manifest = build_manifest(
        csv_name=csv_name,
        full=full,
        normalization=normalization,
        emg_event_k=emg_event_k,
        feature_set=feature_set,
        per_recording=per_recording,
        git_sha=_git_sha(),
    )
    manifest_path = out_dir / f"features_{timestamp}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote manifest -> %s", manifest_path)

    manifest["_csv_path"] = str(csv_path)
    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def _git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parent),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None
