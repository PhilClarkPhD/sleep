"""
Phase 2 (REM-targeted features) + Phase 3 (HMM/Viterbi decoding) experiments.

Same honesty rules as Phase 1: leave-one-animal-out reporting; any selection
(cost-sensitive weight) chosen by inner LOO over training animals only; all
extra features are label-free and computed WITHIN a recording; the HMM
transition matrix is estimated from TRAINING animals only, per outer fold.

Pipelines compared (all pooled LOO):
  1. baseline    v2, w=1, argmax, rule-filter
  2. phase1      v2, cost-sensitive (nested), rule-filter
  3. p2-feats    v2+extra, w=1, argmax, rule-filter
  4. p3-hmm      v2, w=1, HMM decode
  5. p2+p3       v2+extra, w=1, HMM decode
  6. combined    v2+extra, cost-sensitive (nested), HMM decode

Usage:
    python -m model.experiments.rem_phase23_experiment --v2 <fs_v2.csv>
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from sleep_features import apply_rule_based_filter
from sleep_features.config import FEATURE_SETS, UNSCORED_LABEL
from model.diagnostics import evaluate_model
from model.experiments.rem_recall_experiment import BASE_PARAMS, _fit, GROUP_COL, TARGET_COL, TIME_COL, REM

EPS = 1e-9
PRECISION_FLOOR = 0.85
WEIGHT_GRID = [1, 2, 3, 5]


# ------------------------------------------------------------- Phase 2 features
def augment_features(df):
    """Add label-free, within-recording REM-discriminating features.

    NOTE: rolling features are centered (use neighbouring epochs) — a look-ahead
    that is fine for OFFLINE batch scoring and identical at train/serve; they are
    computed per recording so never cross animals.
    """
    df = df.copy()
    extra = []

    # Instantaneous spectral ratios (gain-invariant).
    df["theta_dominance"] = df["theta_rel"] / (df["delta_rel"] + df["theta_rel"] + EPS)
    df["theta_over_sigma"] = df["theta_rel"] / (df["sigma_rel"] + EPS)
    extra += ["theta_dominance", "theta_over_sigma"]

    parts = []
    for _, g in df.groupby(GROUP_COL, sort=False):
        g = g.sort_values(TIME_COL).copy()
        # EMG atonia relative to this recording's own active (p90) level.
        ref = np.quantile(g["EMG_std"].to_numpy(), 0.90) + EPS
        g["emg_atonia"] = g["EMG_std"] / ref
        # Longer temporal context — bout-scale (REM bouts ~1-2 min).
        for col in ["theta_over_delta", "EMG_std", "delta_rel", "theta_dominance"]:
            g[f"{col}_roll5"] = g[col].rolling(5, center=True, min_periods=1).mean()
            g[f"{col}_roll9"] = g[col].rolling(9, center=True, min_periods=1).mean()
        parts.append(g)
    df = pd.concat(parts).sort_index()

    extra += ["emg_atonia"]
    for col in ["theta_over_delta", "EMG_std", "delta_rel", "theta_dominance"]:
        extra += [f"{col}_roll5", f"{col}_roll9"]
    return df, extra


# ------------------------------------------------------------- Phase 3 HMM
def _hmm_params(train_df, classes, alpha=1.0):
    """Priors, log start, log transition matrix from TRAINING label sequences."""
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    trans = np.full((K, K), alpha)
    prior = np.full(K, alpha)
    for _, g in train_df.groupby(GROUP_COL, sort=False):
        seq = g.sort_values(TIME_COL)[TARGET_COL].map(idx).to_numpy()
        for s in seq:
            prior[s] += 1
        for a, b in zip(seq[:-1], seq[1:]):
            trans[a, b] += 1
    prior = prior / prior.sum()
    trans = trans / trans.sum(axis=1, keepdims=True)
    start = prior.copy()
    return prior, np.log(start), np.log(trans)


def _viterbi(log_emission, log_start, log_trans):
    """Standard Viterbi. log_emission: (T, K). Returns state index path (T,)."""
    T, K = log_emission.shape
    dp = np.empty((T, K))
    bp = np.empty((T, K), dtype=int)
    dp[0] = log_start + log_emission[0]
    for t in range(1, T):
        scores = dp[t - 1][:, None] + log_trans  # (K_prev, K_next)
        bp[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + log_emission[t]
    path = np.empty(T, dtype=int)
    path[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def _hmm_decode(proba, prior, log_start, log_trans, classes):
    """Scaled-likelihood emission = posterior / prior, then Viterbi."""
    like = np.clip(proba, 1e-12, None) / prior[None, :]
    path = _viterbi(np.log(like), log_start, log_trans)
    return classes[path]


# ------------------------------------------------------------- LOO runners
def _apply_decoder(proba, mult, rem_idx, classes, decoder, hmm=None):
    p = proba.copy()
    p[:, rem_idx] *= mult
    if decoder == "none":
        return classes[p.argmax(axis=1)]
    if decoder == "rule":
        pred = classes[p.argmax(axis=1)]
        return np.asarray(apply_rule_based_filter(pred))
    elif decoder == "hmm":
        prior, log_start, log_trans = hmm
        return _hmm_decode(p, prior, log_start, log_trans, classes)
    raise ValueError(decoder)


def loo_run(df, cols, weight=1, mult=1.0, decoder="rule"):
    """Plain LOO at fixed (weight, mult, decoder). Returns pooled (y_true, y_pred)."""
    enc = LabelEncoder().fit(df[TARGET_COL])
    rem_idx = list(enc.classes_).index(REM)
    animals = list(df[GROUP_COL].unique())
    yt, yp, grp = [], [], []
    for a in animals:
        tr = df.loc[df[GROUP_COL] != a]
        sub = df.loc[df[GROUP_COL] == a].sort_values(TIME_COL)
        rem_mask = (tr[TARGET_COL] == REM).to_numpy()
        model = _fit(tr[cols], enc.transform(tr[TARGET_COL]), rem_mask, weight)
        proba = model.predict_proba(sub[cols])
        hmm = _hmm_params(tr, enc.classes_) if decoder == "hmm" else None
        pred = _apply_decoder(proba, mult, rem_idx, enc.classes_, decoder, hmm)
        yt.append(sub[TARGET_COL].to_numpy())
        yp.append(pred)
        grp.append(np.full(len(sub), a))
    return np.concatenate(yt), np.concatenate(yp), np.concatenate(grp)


def _rem_pr(m):
    for k, v in m.class_metrics.items():
        if k.lower() == "rem":
            return v["recall"], v["precision"]
    return (np.nan, np.nan)


def loo_run_nested(df, cols, decoder="hmm"):
    """Nested LOO: inner-LOO selects REM weight (mult=1) maximizing inner REM
    recall s.t. inner REM precision >= floor, using the SAME decoder; then apply
    to the held-out outer animal."""
    from sklearn.metrics import precision_score, recall_score

    enc = LabelEncoder().fit(df[TARGET_COL])
    rem_idx = list(enc.classes_).index(REM)
    animals = list(df[GROUP_COL].unique())
    yt, yp, grp, chosen = [], [], [], {}

    for test_animal in animals:
        train_animals = [a for a in animals if a != test_animal]
        # inner LOO over train_animals to pick weight
        best = None
        for w in WEIGHT_GRID:
            iy, ip = [], []
            for inner_test in train_animals:
                inner_tr = df.loc[df[GROUP_COL].isin(
                    [x for x in train_animals if x != inner_test])]
                isub = df.loc[df[GROUP_COL] == inner_test].sort_values(TIME_COL)
                rem_mask = (inner_tr[TARGET_COL] == REM).to_numpy()
                m = _fit(inner_tr[cols], enc.transform(inner_tr[TARGET_COL]), rem_mask, w)
                proba = m.predict_proba(isub[cols])
                hmm = _hmm_params(inner_tr, enc.classes_) if decoder == "hmm" else None
                pred = _apply_decoder(proba, 1.0, rem_idx, enc.classes_, decoder, hmm)
                iy.append(isub[TARGET_COL].to_numpy()); ip.append(pred)
            iy, ip = np.concatenate(iy), np.concatenate(ip)
            rr = recall_score(iy, ip, labels=[REM], average="micro", zero_division=0)
            rp = precision_score(iy, ip, labels=[REM], average="micro", zero_division=0)
            key = (rp >= PRECISION_FLOOR, rr, rp)
            if best is None or key > best[0]:
                best = (key, w)
        w = best[1]
        chosen[test_animal] = w
        tr = df.loc[df[GROUP_COL].isin(train_animals)]
        sub = df.loc[df[GROUP_COL] == test_animal].sort_values(TIME_COL)
        rem_mask = (tr[TARGET_COL] == REM).to_numpy()
        model = _fit(tr[cols], enc.transform(tr[TARGET_COL]), rem_mask, w)
        proba = model.predict_proba(sub[cols])
        hmm = _hmm_params(tr, enc.classes_) if decoder == "hmm" else None
        pred = _apply_decoder(proba, 1.0, rem_idx, enc.classes_, decoder, hmm)
        yt.append(sub[TARGET_COL].to_numpy()); yp.append(pred)
        grp.append(np.full(len(sub), test_animal))
    return np.concatenate(yt), np.concatenate(yp), np.concatenate(grp), chosen


def _load(path):
    df = pd.read_csv(path)
    df = df.loc[df[TARGET_COL] != UNSCORED_LABEL].copy()
    df = df.sort_values([GROUP_COL, TIME_COL]).reset_index(drop=True)
    cols = [c for c in FEATURE_SETS["v2"] if c in df.columns]
    return df, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--skip-hmm-nested", action="store_true",
                    help="Skip the (slow) nested cost-sensitive + HMM pipeline")
    args = ap.parse_args()

    df, v2_cols = _load(args.v2)
    df_aug, extra = augment_features(df)
    aug_cols = v2_cols + extra
    print(f"Loaded {len(df)} rows, v2={len(v2_cols)} feats, +{len(extra)} extra = {len(aug_cols)}")
    print(f"Extra features: {extra}")

    results = []
    per_animal = {}  # name -> {animal: (REM recall, REM precision, kappa)}

    def record(name, yt, yp, grp, note=""):
        m = evaluate_model(yt, yp, analyze_transitions=True)
        rr, rp = _rem_pr(m)
        results.append((name, rr, rp, m.cohen_kappa, m.balanced_accuracy,
                        m.f1_macro, m.transition_accuracy, note))
        print(f"  [{name}] REM rec {rr:.3f} prec {rp:.3f} | kappa {m.cohen_kappa:.3f} "
              f"balAcc {m.balanced_accuracy:.3f} macroF1 {m.f1_macro:.3f}", flush=True)
        rows = {}
        for a in np.unique(grp):
            sel = grp == a
            ma = evaluate_model(yt[sel], yp[sel], analyze_transitions=False)
            rows[str(a)] = (*_rem_pr(ma), ma.cohen_kappa)
        per_animal[name] = rows

    print("\n1. baseline (v2, w1, rule)", flush=True)
    record("baseline", *loo_run(df_aug, v2_cols, 1, 1.0, "rule"))

    print("1b. baseline-none (v2, w1, no filter)", flush=True)
    record("baseline-none", *loo_run(df_aug, v2_cols, 1, 1.0, "none"))

    print("2. p2-feats (v2+extra, w1, rule)", flush=True)
    record("p2-feats", *loo_run(df_aug, aug_cols, 1, 1.0, "rule"))

    print("2b. p2-feats-none (v2+extra, w1, no filter)", flush=True)
    record("p2-feats-none", *loo_run(df_aug, aug_cols, 1, 1.0, "none"))

    print("3. p3-hmm (v2, w1, hmm)", flush=True)
    record("p3-hmm", *loo_run(df_aug, v2_cols, 1, 1.0, "hmm"))

    print("4. p2+p3 (v2+extra, w1, hmm)", flush=True)
    record("p2+p3", *loo_run(df_aug, aug_cols, 1, 1.0, "hmm"))

    print("5. combined-none (v2+extra, cost-sensitive nested, no filter)  <- recommended serving config", flush=True)
    yt, yp, grp, chosen = loo_run_nested(df_aug, aug_cols, "none")
    record("combined-none", yt, yp, grp, note=f"weights={chosen}")
    print(f"     selected weights: {chosen}", flush=True)

    print("6. combined-rule (v2+extra, cost-sensitive nested, rule)", flush=True)
    yt, yp, grp, chosen = loo_run_nested(df_aug, aug_cols, "rule")
    record("combined-rule", yt, yp, grp, note=f"weights={chosen}")
    print(f"     selected weights: {chosen}", flush=True)

    if not args.skip_hmm_nested:
        print("7. combined-hmm (v2+extra, cost-sensitive nested, hmm)", flush=True)
        yt, yp, grp, chosen = loo_run_nested(df_aug, aug_cols, "hmm")
        record("combined-hmm", yt, yp, grp, note=f"weights={chosen}")
        print(f"     selected weights: {chosen}", flush=True)

    print("\n" + "#" * 78)
    print("SUMMARY — leave-one-animal-out (REM recall is the target)")
    print("#" * 78)
    hdr = f"{'pipeline':<16}{'REMrec':>8}{'REMprec':>8}{'kappa':>8}{'balAcc':>8}{'macroF1':>8}{'trans':>8}"
    print(hdr)
    for name, rr, rp, k, ba, mf, tr, note in results:
        print(f"{name:<16}{rr:>8.3f}{rp:>8.3f}{k:>8.3f}{ba:>8.3f}{mf:>8.3f}{tr:>8.3f}")

    print("\nPER-ANIMAL (REM recall / REM precision / kappa), held-out animal in each column")
    animals = sorted(next(iter(per_animal.values())).keys())
    for name in ("baseline-none", "p2-feats-none", "combined-none"):
        print(f"\n  {name}")
        for a in animals:
            rr, rp, k = per_animal[name][a]
            print(f"    {a:<16} REM rec {rr:.2f}  prec {rp:.2f}  kappa {k:.2f}")


if __name__ == "__main__":
    main()
