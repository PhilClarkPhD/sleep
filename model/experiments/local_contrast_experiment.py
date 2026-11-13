"""
Local-contrast features for short REM bouts.

Singleton / short REM epochs have a weaker absolute theta signature (theta/delta
~1.4 vs ~2.7 in sustained bouts) but still a ~2x jump relative to the NREM that
surrounds them. Cohort-level features compare an epoch to REM everywhere; a human
compares it to the preceding minutes. This adds that comparison:

    <col>_lc<W> = col / running_median(col over the PREVIOUS W epochs)

The median is trailing (causal) and excludes the current epoch, so a one-epoch
event cannot dilute its own baseline — unlike the centered rolling means already
in the feature set, which do exactly that.

Evaluated as pooled leave-one-animal-out, v2 + REM extras, floor-ceiling EMG
scaling, unweighted, no filter; recall is reported by REM bout length.

Usage:
    python -m model.experiments.local_contrast_experiment --v2 <fs_v2.csv>
"""

import argparse

import numpy as np
import pandas as pd

from sleep_features.config import FEATURE_SETS, UNSCORED_LABEL
from model.diagnostics import evaluate_model
from model.experiments.diagnose_animals import _bouts, loo_predictions
from model.experiments.emg_scaling_experiment import apply_scaling
from model.experiments.rem_phase23_experiment import augment_features
from model.experiments.rem_recall_experiment import GROUP_COL, REM, TARGET_COL, TIME_COL

EPS = 1e-9
LC_SOURCE = ["theta_over_delta", "theta_dominance", "theta_rel"]


def add_local_contrast(df: pd.DataFrame, windows=(30,), cols=LC_SOURCE):
    """Append trailing-median local-contrast columns; returns (df, new_col_names)."""
    df = df.copy()
    new = []
    parts = []
    for _, g in df.groupby(GROUP_COL, sort=False):
        g = g.sort_values(TIME_COL).copy()
        for col in cols:
            s = g[col]
            for w in windows:
                base = s.shift(1).rolling(w, min_periods=max(5, w // 3)).median()
                base = base.bfill()  # first epochs: use the earliest available baseline
                g[f"{col}_lc{w}"] = np.log((s + EPS) / (base + EPS))
        parts.append(g)
    df = pd.concat(parts).sort_index()
    for col in cols:
        for w in windows:
            new.append(f"{col}_lc{w}")
    return df, new


def summarize(tag, preds):
    m = evaluate_model(preds["y"].to_numpy(), preds["pred"].to_numpy(), analyze_transitions=True)
    rem = next(v for k, v in m.class_metrics.items() if k.lower() == "rem")
    cat = {"1": [], "2-3": [], "4-9": [], "10+": []}
    per = {}
    for a, s in preds.groupby(GROUP_COL, sort=False):
        y, p = s["y"].to_numpy(), s["pred"].to_numpy()
        for st, en in _bouts(y == REM):
            L = en - st + 1
            k = "1" if L == 1 else "2-3" if L <= 3 else "4-9" if L <= 9 else "10+"
            cat[k].append((p[st:en + 1] == REM).mean())
        ma = evaluate_model(y, p, analyze_transitions=False)
        per[a.split("_")[0]] = next(v for k, v in ma.class_metrics.items() if k.lower() == "rem")["recall"]
    print(f"[{tag}]")
    print(f"    pooled REM rec {rem['recall']:.3f} prec {rem['precision']:.3f} | kappa {m.cohen_kappa:.3f} "
          f"balAcc {m.balanced_accuracy:.3f} trans {m.transition_accuracy:.3f}")
    print("    recall by bout length  " + "  ".join(f"{k}:{np.mean(v):.2f}" for k, v in cat.items() if v))
    print("    per-animal recall      " + "  ".join(f"{a}:{v:.2f}" for a, v in per.items()), flush=True)
    return rem["recall"], rem["precision"], m.cohen_kappa, {k: np.mean(v) for k, v in cat.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    args = ap.parse_args()

    base = pd.read_csv(args.v2)
    base = base.loc[base[TARGET_COL] != UNSCORED_LABEL].copy()
    base = base.sort_values([GROUP_COL, TIME_COL]).reset_index(drop=True)
    v2 = [c for c in FEATURE_SETS["v2"] if c in base.columns]

    df, extra = augment_features(apply_scaling(base, "floor_ceiling"))
    df, lc = add_local_contrast(df, windows=(30, 90))
    lc30 = [c for c in lc if c.endswith("_lc30")]
    lc90 = [c for c in lc if c.endswith("_lc90")]
    roll = [c for c in extra if c.endswith("_roll5") or c.endswith("_roll9")]
    basecols = v2 + extra

    results = {}
    results["baseline (floor-ceiling EMG)"] = summarize("baseline (floor-ceiling EMG)", loo_predictions(df, basecols))
    results["+ lc30 (5 min)"] = summarize("+ lc30 (5 min)", loo_predictions(df, basecols + lc30))
    results["+ lc90 (15 min)"] = summarize("+ lc90 (15 min)", loo_predictions(df, basecols + lc90))
    results["+ lc30 + lc90"] = summarize("+ lc30 + lc90", loo_predictions(df, basecols + lc30 + lc90))
    results["lc30+lc90 replacing roll5/roll9"] = summarize(
        "lc30+lc90 replacing roll5/roll9", loo_predictions(df, [c for c in basecols if c not in roll] + lc30 + lc90))

    print("\n" + "#" * 78)
    print(f"{'variant':<36}{'REMrec':>8}{'REMprec':>8}{'kappa':>8}{'single':>8}{'2-3':>8}{'4-9':>8}{'10+':>8}")
    for k, (rr, rp, kp, bl) in results.items():
        print(f"{k:<36}{rr:>8.3f}{rp:>8.3f}{kp:>8.3f}{bl['1']:>8.2f}{bl['2-3']:>8.2f}{bl['4-9']:>8.2f}{bl['10+']:>8.2f}")


if __name__ == "__main__":
    main()
