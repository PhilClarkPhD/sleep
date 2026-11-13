"""
Temporal-context features.

Sleep state is highly autocorrelated, but the base feature table treats each
epoch independently — the only temporal information currently used is the
post-hoc rule-based filter. These features give the model temporal context
directly: a centered rolling mean (local smoothing) and a first difference
(rate of change) of a few core features.

Computed per recording, so they must be applied to one recording's table at a
time (which is how the builder and serving operate).
"""

from typing import List, Sequence

import pandas as pd

# Core features whose local trend/change is most informative for state.
DEFAULT_CONTEXT_COLS = ["delta_rel", "theta_over_delta", "EMG_std", "sigma_rel"]


def temporal_feature_names(cols: Sequence[str] = DEFAULT_CONTEXT_COLS) -> List[str]:
    names: List[str] = []
    for c in cols:
        names.append(f"{c}_roll3")
        names.append(f"{c}_diff1")
    return names


def add_temporal_context(
    table: pd.DataFrame,
    cols: Sequence[str] = DEFAULT_CONTEXT_COLS,
    window: int = 3,
) -> pd.DataFrame:
    """
    Append rolling-mean and first-difference columns for ``cols``.

    Edge epochs (where the window/diff is undefined) are filled to keep the
    table free of NaN: rolling uses ``min_periods=1``; the diff's first row is 0.
    """
    out = table.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[f"{c}_roll3"] = (
            out[c].rolling(window=window, center=True, min_periods=1).mean()
        )
        out[f"{c}_diff1"] = out[c].diff().fillna(0.0)
    return out
