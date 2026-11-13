"""
Robust parsing of hand-scored sleep-state files.

Two formats are seen in the wild and both are supported:

1. "simple"  — header ``epoch,score``; epoch is 0-indexed.
       epoch,score
       0,Wake
       1,Wake

2. "lab"     — LabChart-style export; header
       ``Epoch #,Start Time,End Time,Score #, Score``; epoch is 1-indexed and
       trailing truncated lines may appear.

Both are normalized to a 0-indexed ``pandas.Series`` named ``score`` so the
feature-store builder can align scores to feature epochs by integer position.
"""

import csv
from pathlib import Path
from typing import Union

import pandas as pd

from .config import UNSCORED_LABEL


def _norm(name: str) -> str:
    return name.strip().lower()


def load_scores(path: Union[str, Path]) -> pd.Series:
    """
    Parse a scores file into a 0-indexed Series of state labels.

    Returns:
        pd.Series indexed by 0-based epoch, values are state-label strings.

    Raises:
        ValueError: if neither a recognizable epoch nor score column is found.
    """
    path = Path(path)

    # Sniff the header to locate the epoch and score columns by name.
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    cols = {_norm(c): i for i, c in enumerate(header)}

    epoch_key = next((k for k in cols if k in ("epoch", "epoch #", "epoch#")), None)
    score_key = next((k for k in cols if k == "score"), None)
    if score_key is None:
        raise ValueError(f"No 'score' column found in {path} (header={header})")

    # Read with bad-line tolerance for truncated trailing rows.
    df = pd.read_csv(
        path,
        usecols=[cols[k] for k in ([epoch_key, score_key] if epoch_key else [score_key])],
        engine="python",
        on_bad_lines="skip",
    )
    df.columns = [_norm(c) for c in df.columns]

    # Drop rows with no label on the *raw* value (catches NaN/None from
    # truncated trailing lines) before stringifying.
    has_label = df[score_key].notna()
    scores = df[score_key].where(has_label).astype("string").str.strip()
    has_label = has_label & scores.ne("") & scores.notna()

    if epoch_key is not None:
        epochs = pd.to_numeric(df[epoch_key], errors="coerce")
        valid = epochs.notna() & has_label
        epochs = epochs[valid].astype(int)
        scores = scores[valid].astype(str)
        # "lab" format is 1-indexed; normalize so the minimum epoch becomes 0.
        if len(epochs) and epochs.min() == 1:
            epochs = epochs - 1
        out = pd.Series(scores.values, index=epochs.values, name="score")
        out = out[~out.index.duplicated(keep="first")].sort_index()
    else:
        # No epoch column: rows are implicitly 0..n-1 in file order.
        scores = scores[has_label].astype(str).reset_index(drop=True)
        out = pd.Series(scores.values, name="score")
        out.index.name = None

    return out


def align_scores(scores: pd.Series, n_epochs: int) -> pd.Series:
    """
    Align a parsed score Series to ``n_epochs`` feature rows (0..n_epochs-1).

    Epochs with no label become ``UNSCORED_LABEL`` so they can be dropped
    downstream rather than silently misaligned.
    """
    aligned = scores.reindex(range(n_epochs))
    aligned = aligned.fillna(UNSCORED_LABEL)
    aligned.name = "score"
    return aligned
