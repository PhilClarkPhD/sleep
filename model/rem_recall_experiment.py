"""
REM-recall improvement experiment — honest, leakage-free.

Phase 0: size the ceiling. Plain leave-one-animal-out; collect REM one-vs-rest
         predict_proba and report the recall reachable at precision floors by
         THRESHOLD ALONE (before any refit).

Phase 1: cost-sensitive operating point via NESTED leave-one-animal-out.
         Outer 5-fold LOO is the only thing scored. For each outer fold, an
         inner LOO over the 4 TRAINING animals selects (REM sample-weight,
         REM probability-multiplier) to maximize REM recall subject to a REM
         precision floor. The held-out outer animal never influences its own
         operating point -> no selection leakage.

Usage:
    python -m model.rem_recall_experiment --v2 <fs_v2.csv>
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from sleep_features import apply_rule_based_filter
from sleep_features.config import FEATURE_SETS, UNSCORED_LABEL
from model.diagnostics import evaluate_model

GROUP_COL, TARGET_COL, TIME_COL = "ID_day", "score", "epoch"
REM = "REM"
PRECISION_FLOOR = 0.85

BASE_PARAMS = dict(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
)
WEIGHT_GRID = [1, 2, 3, 5]          # REM sample-weight multiplier
MULT_GRID = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]  # REM predict_proba multiplier


def _fit(X, y_enc, rem_mask, weight):
    sw = np.ones(len(y_enc))
    sw[rem_mask] = weight
    m = XGBClassifier(**BASE_PARAMS)
    m.fit(X, y_enc, sample_weight=sw)
    return m


def _predict(model, X, rem_idx, mult, classes):
    """argmax after scaling the REM probability column by `mult`."""
    proba = model.predict_proba(X).copy()
    proba[:, rem_idx] *= mult
    return classes[proba.argmax(axis=1)]


def _load(path, feature_set="v2"):
    df = pd.read_csv(path)
    df = df.loc[df[TARGET_COL] != UNSCORED_LABEL].copy()
    df = df.sort_values([GROUP_COL, TIME_COL]).reset_index(drop=True)
    cols = [c for c in FEATURE_SETS[feature_set] if c in df.columns]
    return df, cols


# ---------------------------------------------------------------- Phase 0
def phase0(df, cols):
    enc = LabelEncoder().fit(df[TARGET_COL])
    rem_idx = list(enc.classes_).index(REM)
    animals = list(df[GROUP_COL].unique())

    y_true, p_rem = [], []
    for a in animals:
        te = df[GROUP_COL] == a
        tr = df.loc[~te]
        model = XGBClassifier(**BASE_PARAMS)
        model.fit(tr[cols], enc.transform(tr[TARGET_COL]))
        proba = model.predict_proba(df.loc[te, cols])
        p_rem.append(proba[:, rem_idx])
        y_true.append((df.loc[te, TARGET_COL] == REM).to_numpy().astype(int))
    y_true = np.concatenate(y_true)
    p_rem = np.concatenate(p_rem)

    prec, rec, thr = precision_recall_curve(y_true, p_rem)
    ap = average_precision_score(y_true, p_rem)

    def recall_at(pfloor):
        ok = prec >= pfloor
        if not ok.any():
            return (np.nan, np.nan)
        # among points meeting the precision floor, take the max recall
        i = np.argmax(np.where(ok, rec, -1))
        t = thr[min(i, len(thr) - 1)]
        return rec[i], t

    print("\n" + "=" * 60)
    print("PHASE 0 — REM one-vs-rest PR headroom (raw proba, pre-filter)")
    print("=" * 60)
    print(f"  REM average precision (AP): {ap:.4f}")
    for pf in (0.90, 0.85, 0.80):
        r, t = recall_at(pf)
        print(f"  recall @ precision>={pf:.2f}:  {r:.4f}  (REM-proba thr ~{t:.3f})")
    print(f"  [baseline argmax operating point had recall 0.648 @ precision 0.916]")


# ---------------------------------------------------------------- Phase 1
def _loo_pooled(df, cols, animals, enc, rem_idx, weight, mult, use_filter=True):
    """Pooled predictions from LOO over `animals`, at a fixed (weight, mult)."""
    yt, yp = [], []
    for a in animals:
        te = df[GROUP_COL] == a
        tr = df.loc[df[GROUP_COL].isin([x for x in animals if x != a])]
        rem_mask = (tr[TARGET_COL] == REM).to_numpy()
        model = _fit(tr[cols], enc.transform(tr[TARGET_COL]), rem_mask, weight)
        sub = df.loc[te].sort_values(TIME_COL)
        pred = _predict(model, sub[cols], rem_idx, mult, enc.classes_)
        if use_filter:
            pred = np.asarray(apply_rule_based_filter(pred))
        yt.append(sub[TARGET_COL].to_numpy())
        yp.append(pred)
    return np.concatenate(yt), np.concatenate(yp)


def _select_operating_point(df, cols, train_animals, enc, rem_idx):
    """Inner LOO over train_animals: pick (weight, mult) maximizing REM recall
    subject to REM precision >= floor. Trains once per weight, sweeps mult free."""
    # Pre-fit inner models: one per (held-in inner-test animal, weight).
    best = None  # (recall, precision, weight, mult)
    # Cache predictions per (inner_test, weight) -> (proba, y_true_sub)
    cache = {}
    for inner_test in train_animals:
        inner_tr = [x for x in train_animals if x != inner_test]
        tr = df.loc[df[GROUP_COL].isin(inner_tr)]
        sub = df.loc[df[GROUP_COL] == inner_test].sort_values(TIME_COL)
        rem_mask = (tr[TARGET_COL] == REM).to_numpy()
        for w in WEIGHT_GRID:
            model = _fit(tr[cols], enc.transform(tr[TARGET_COL]), rem_mask, w)
            cache[(inner_test, w)] = (
                model.predict_proba(sub[cols]),
                sub[TARGET_COL].to_numpy(),
            )
    for w, mult in itertools.product(WEIGHT_GRID, MULT_GRID):
        yt, yp = [], []
        for inner_test in train_animals:
            proba, y_sub = cache[(inner_test, w)]
            p = proba.copy()
            p[:, rem_idx] *= mult
            pred = enc.classes_[p.argmax(axis=1)]
            pred = np.asarray(apply_rule_based_filter(pred))
            yt.append(y_sub)
            yp.append(pred)
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        rem_rec = recall_score(yt, yp, labels=[REM], average="micro", zero_division=0)
        rem_prec = precision_score(yt, yp, labels=[REM], average="micro", zero_division=0)
        # objective: precision floor first, then max recall, then max precision
        meets = rem_prec >= PRECISION_FLOOR
        key = (meets, rem_rec, rem_prec)
        if best is None or key > best[0]:
            best = (key, w, mult, rem_rec, rem_prec)
    _, w, mult, _, _ = best
    return w, mult


def phase1(df, cols):
    enc = LabelEncoder().fit(df[TARGET_COL])
    rem_idx = list(enc.classes_).index(REM)
    animals = list(df[GROUP_COL].unique())

    # Baseline (weight=1, mult=1) under the SAME pipeline, for apples-to-apples.
    print("\n" + "=" * 60)
    print("PHASE 1 — nested-LOO cost-sensitive operating point")
    print("=" * 60)
    bt, bp = _loo_pooled(df, cols, animals, enc, rem_idx, weight=1, mult=1.0)
    base = evaluate_model(bt, bp, analyze_transitions=True)

    # Nested: choose operating point on inner animals, apply to outer test animal.
    yt, yp, chosen = [], [], {}
    for test_animal in animals:
        train_animals = [a for a in animals if a != test_animal]
        w, mult = _select_operating_point(df, cols, train_animals, enc, rem_idx)
        chosen[test_animal] = (w, mult)
        tr = df.loc[df[GROUP_COL].isin(train_animals)]
        rem_mask = (tr[TARGET_COL] == REM).to_numpy()
        model = _fit(tr[cols], enc.transform(tr[TARGET_COL]), rem_mask, w)
        sub = df.loc[df[GROUP_COL] == test_animal].sort_values(TIME_COL)
        pred = _predict(model, sub[cols], rem_idx, mult, enc.classes_)
        pred = np.asarray(apply_rule_based_filter(pred))
        yt.append(sub[TARGET_COL].to_numpy())
        yp.append(pred)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    tuned = evaluate_model(yt, yp, analyze_transitions=True)

    def rem_pr(m):
        cm = m.class_metrics
        for k, v in cm.items():
            if k.lower() == "rem":
                return v["recall"], v["precision"]
        return (np.nan, np.nan)

    br, bpx = rem_pr(base)
    tr_, tpx = rem_pr(tuned)
    print("\nSelected (REM weight, REM proba-mult) per outer fold:")
    for a, (w, mult) in chosen.items():
        print(f"  {a:20s} weight={w} mult={mult}")

    print("\n" + "-" * 60)
    print(f"{'metric':<16}{'baseline':>12}{'tuned':>12}{'delta':>10}")
    print("-" * 60)
    rows = [
        ("REM recall", br, tr_),
        ("REM precision", bpx, tpx),
        ("Cohen kappa", base.cohen_kappa, tuned.cohen_kappa),
        ("balanced acc", base.balanced_accuracy, tuned.balanced_accuracy),
        ("F1 macro", base.f1_macro, tuned.f1_macro),
        ("F1 weighted", base.f1_weighted, tuned.f1_weighted),
        ("transition acc", base.transition_accuracy, tuned.transition_accuracy),
    ]
    for name, b, t in rows:
        print(f"{name:<16}{b:>12.4f}{t:>12.4f}{t-b:>+10.4f}")

    print("\nGuardrails: REM precision floor", PRECISION_FLOOR,
          "| kappa not worse")
    ok_prec = tpx >= PRECISION_FLOOR
    ok_kappa = tuned.cohen_kappa >= base.cohen_kappa - 1e-9
    ok_recall = tr_ > br
    print(f"  REM recall improved:   {ok_recall}  ({br:.3f} -> {tr_:.3f})")
    print(f"  REM precision >= {PRECISION_FLOOR}: {ok_prec}  ({tpx:.3f})")
    print(f"  kappa not worse:       {ok_kappa}  ({base.cohen_kappa:.3f} -> {tuned.cohen_kappa:.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v2", required=True, help="Path to v2 feature-store CSV")
    args = p.parse_args()
    df, cols = _load(args.v2, "v2")
    print(f"Loaded {len(df)} scored rows, {len(cols)} features, "
          f"{df[GROUP_COL].nunique()} animals")
    phase0(df, cols)
    phase1(df, cols)


if __name__ == "__main__":
    main()
