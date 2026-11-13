"""
EMG scaling experiment — can a shape-aligning per-recording transform of the EMG
amplitude features fix animals whose EMG dynamic range is compressed (ZZ05)?

Per-recording robust_mad is affine, so it cancels gain/offset but not a
compressed Wake→REM contrast (additive noise floor). Two alternatives that align
the distribution *shape* across recordings are tested on top of the v2 store:

  mad           current: (x - median) / MAD                      [baseline]
  floor_ceiling (x - p_lo) / (p_hi - p_lo), default p05/p95       [range-anchored]
  rank          percentile rank within the recording, in [0, 1]   [shape-aligned]

Both are invariant to any affine transform of x, so computing them on the
robust_mad-normalized store is identical to computing them on raw amplitudes.
Transforms are applied to the EMG amplitude features (EMG_std, EMG_ptp); the
temporal-context columns derived from EMG_std and the REM extras (emg_atonia,
EMG_std_roll*) are recomputed from the transformed values.

Evaluation: pooled leave-one-animal-out, v2 + REM extras, no filter, REM weight
1 (fast) and optionally the nested cost-sensitive selection for the best variant.

Usage:
    python -m model.emg_scaling_experiment --v2 <fs_v2.csv> [--nested]
"""

import argparse

import numpy as np
import pandas as pd

from sleep_features.config import FEATURE_SETS, UNSCORED_LABEL
from model.diagnostics import evaluate_model
from model.rem_phase23_experiment import augment_features, loo_run, loo_run_nested
from model.rem_recall_experiment import GROUP_COL, REM, TARGET_COL, TIME_COL

EMG_AMP = ["EMG_std", "EMG_ptp"]
CONTEXT_SRC = "EMG_std"  # temporal context columns EMG_std_roll3 / EMG_std_diff1


def _floor_ceiling(x: np.ndarray, lo=5, hi=95) -> np.ndarray:
    a, b = np.nanpercentile(x, [lo, hi])
    return (x - a) / ((b - a) + 1e-12)


def _rank(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(pct=True, method="average").to_numpy()
    return r


def apply_scaling(df: pd.DataFrame, method: str, lo=5, hi=95) -> pd.DataFrame:
    """Return a copy of df with EMG amplitude features re-scaled per recording."""
    if method == "mad":
        return df
    out = df.copy()
    for _, idx in out.groupby(GROUP_COL, sort=False).groups.items():
        g = out.loc[idx].sort_values(TIME_COL)
        for col in EMG_AMP:
            x = g[col].to_numpy(dtype=float)
            if method == "floor_ceiling":
                y = _floor_ceiling(x, lo, hi)
            elif method == "rank":
                y = _rank(x)
            else:
                raise ValueError(method)
            out.loc[g.index, col] = y
        # Recompute temporal context from the transformed EMG_std (same
        # definitions as sleep_features.temporal: centered roll3 mean, diff1).
        s = pd.Series(out.loc[g.index, CONTEXT_SRC].to_numpy(), index=g.index)
        if f"{CONTEXT_SRC}_roll3" in out:
            out.loc[g.index, f"{CONTEXT_SRC}_roll3"] = s.rolling(3, center=True, min_periods=1).mean().to_numpy()
        if f"{CONTEXT_SRC}_diff1" in out:
            out.loc[g.index, f"{CONTEXT_SRC}_diff1"] = s.diff().fillna(0).to_numpy()
    return out


def rem_shift(df: pd.DataFrame, col="EMG_std") -> pd.Series:
    """z of each animal's REM-epoch median vs the other animals'."""
    med = df.loc[df[TARGET_COL] == REM].groupby(GROUP_COL)[col].median()
    z = {}
    for a in med.index:
        o = med.drop(index=a)
        z[a.split("_")[0]] = (med[a] - o.mean()) / (o.std(ddof=1) + 1e-9)
    return pd.Series(z)


def per_animal(yt, yp, grp):
    rows = {}
    for a in np.unique(grp):
        m = evaluate_model(yt[grp == a], yp[grp == a], analyze_transitions=False)
        rem = next(v for k, v in m.class_metrics.items() if k.lower() == "rem")
        rows[a.split("_")[0]] = (rem["recall"], rem["precision"], m.cohen_kappa)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--nested", action="store_true", help="also run nested cost-sensitive per variant")
    ap.add_argument("--variants", nargs="*", default=["mad", "floor_ceiling", "rank"])
    args = ap.parse_args()

    base = pd.read_csv(args.v2)
    base = base.loc[base[TARGET_COL] != UNSCORED_LABEL].copy()
    base = base.sort_values([GROUP_COL, TIME_COL]).reset_index(drop=True)
    v2_cols = [c for c in FEATURE_SETS["v2"] if c in base.columns]

    summary, animals_tbl, shifts = [], {}, {}
    for method in args.variants:
        df = apply_scaling(base, method)
        shifts[method] = rem_shift(df)
        df_aug, extra = augment_features(df)
        cols = v2_cols + extra
        print(f"\n=== {method} ===", flush=True)
        yt, yp, grp = loo_run(df_aug, cols, 1, 1.0, "none")
        m = evaluate_model(yt, yp, analyze_transitions=True)
        rem = next(v for k, v in m.class_metrics.items() if k.lower() == "rem")
        summary.append((method, "w=1", rem["recall"], rem["precision"], m.cohen_kappa, m.balanced_accuracy, m.transition_accuracy))
        animals_tbl[(method, "w=1")] = per_animal(yt, yp, grp)
        print(f"  w=1    REM rec {rem['recall']:.3f} prec {rem['precision']:.3f} kappa {m.cohen_kappa:.3f}", flush=True)
        if args.nested:
            yt, yp, grp, chosen = loo_run_nested(df_aug, cols, "none")
            m = evaluate_model(yt, yp, analyze_transitions=True)
            rem = next(v for k, v in m.class_metrics.items() if k.lower() == "rem")
            summary.append((method, "nested", rem["recall"], rem["precision"], m.cohen_kappa, m.balanced_accuracy, m.transition_accuracy))
            animals_tbl[(method, "nested")] = per_animal(yt, yp, grp)
            print(f"  nested REM rec {rem['recall']:.3f} prec {rem['precision']:.3f} kappa {m.cohen_kappa:.3f} weights={chosen}", flush=True)

    print("\n" + "#" * 72)
    print("REM-epoch EMG_std shift (z vs other animals) after each scaling")
    print(pd.DataFrame(shifts).round(1).to_string())

    print("\n" + "#" * 72)
    print(f"{'scaling':<15}{'mode':<8}{'REMrec':>8}{'REMprec':>8}{'kappa':>8}{'balAcc':>8}{'trans':>8}")
    for s in summary:
        print(f"{s[0]:<15}{s[1]:<8}" + "".join(f"{v:>8.3f}" for v in s[2:]))

    print("\nPer-animal REM recall (precision)")
    keys = list(animals_tbl)
    animals = sorted(next(iter(animals_tbl.values())).keys())
    print(f"{'animal':<8}" + "".join(f"{k[0][:9]+'/'+k[1]:>20}" for k in keys))
    for a in animals:
        print(f"{a:<8}" + "".join(f"{animals_tbl[k][a][0]:>12.2f} ({animals_tbl[k][a][1]:.2f})" for k in keys))


if __name__ == "__main__":
    main()
