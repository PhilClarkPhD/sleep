"""
Leave-one-animal-out (LOO) out-of-sample comparison of feature sets.

Honest cross-subject generalization estimate: for each recording, train on all
OTHER animals and predict the held-out animal. Predictions are pooled across all
folds and scored once. No animal ever appears in both train and test, so this
answers "how well does the model score a brand-new rat?" — unlike the within-
recording temporal holdout in train.py.

Hyperparameters are held FIXED and identical across feature sets so the only
thing that varies between runs is the feature columns (a clean A/B).

Usage:
    python -m model.compare_feature_sets --v1 <fs_v1.csv> --v2 <fs_v2.csv>
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from sleep_features import apply_rule_based_filter
from sleep_features.config import FEATURE_SETS, UNSCORED_LABEL

from model.diagnostics import evaluate_model

# Fixed, shared hyperparameters — identical for every feature set and fold.
FIXED_PARAMS = dict(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
)

GROUP_COL = "ID_day"
TARGET_COL = "score"
TIME_COL = "epoch"


def loo_evaluate(df: pd.DataFrame, feature_cols, use_filter: bool = True):
    """Run leave-one-animal-out CV; return (metrics, per_animal_kappa, pooled_df)."""
    df = df.loc[df[TARGET_COL] != UNSCORED_LABEL].copy()
    df = df.sort_values([GROUP_COL, TIME_COL]).reset_index(drop=True)

    # Global label encoding so class indices are stable across folds.
    encoder = LabelEncoder().fit(df[TARGET_COL])

    animals = list(df[GROUP_COL].unique())
    y_true_all, y_pred_all = [], []
    per_animal = {}

    for held_out in animals:
        test_mask = df[GROUP_COL] == held_out
        train_df = df.loc[~test_mask]
        test_df = df.loc[test_mask].sort_values(TIME_COL)

        X_train = train_df[feature_cols]
        y_train = encoder.transform(train_df[TARGET_COL])
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET_COL].to_numpy()

        model = XGBClassifier(**FIXED_PARAMS)
        model.fit(X_train, y_train)
        y_pred = encoder.inverse_transform(model.predict(X_test))

        if use_filter:
            # Test block is one animal's contiguous timeline -> filter is valid.
            y_pred = np.asarray(apply_rule_based_filter(y_pred))

        per_animal[held_out] = cohen_kappa_score(y_test, y_pred)
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    metrics = evaluate_model(y_true, y_pred, analyze_transitions=True)
    pooled = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    return metrics, per_animal, pooled


def _rem_recall(metrics) -> float:
    for name, m in metrics.class_metrics.items():
        if "rem" in name.lower() and "non" not in name.lower():
            return m["recall"]
    return float("nan")


def _summary_row(label, metrics, per_animal):
    return {
        "feature_set": label,
        "n_features": None,
        "cohen_kappa": round(metrics.cohen_kappa, 4),
        "balanced_acc": round(metrics.balanced_accuracy, 4),
        "f1_weighted": round(metrics.f1_weighted, 4),
        "f1_macro": round(metrics.f1_macro, 4),
        "rem_recall": round(_rem_recall(metrics), 4),
        "transition_acc": round(metrics.transition_accuracy, 4)
        if metrics.transition_accuracy is not None
        else None,
        "worst_animal_kappa": round(min(per_animal.values()), 4),
        "per_animal_kappa": {k: round(v, 4) for k, v in per_animal.items()},
    }


def run(stores: dict):
    rows = []
    for label, path in stores.items():
        feature_cols = FEATURE_SETS[label]
        df = pd.read_csv(path)
        available = [c for c in feature_cols if c in df.columns]
        missing = set(feature_cols) - set(available)
        if missing:
            print(f"WARNING [{label}]: missing columns {sorted(missing)}")
        print(f"\n{'='*70}\nLOO evaluation: {label}  ({len(available)} features, {len(df)} rows)\n{'='*70}")
        metrics, per_animal, _ = loo_evaluate(df, available)
        print(metrics)
        print("\nPer-animal Cohen's kappa:")
        for animal, k in per_animal.items():
            print(f"  {animal:20s} {k:.4f}")
        row = _summary_row(label, metrics, per_animal)
        row["n_features"] = len(available)
        rows.append(row)

    print(f"\n\n{'#'*70}\nSUMMARY — leave-one-animal-out out-of-sample\n{'#'*70}")
    hdr = f"{'set':>4} {'nfeat':>5} {'kappa':>7} {'bal_acc':>8} {'f1_wt':>7} {'f1_mac':>7} {'REM_rec':>8} {'trans':>7} {'worst_k':>8}"
    print(hdr)
    for r in rows:
        print(
            f"{r['feature_set']:>4} {r['n_features']:>5} {r['cohen_kappa']:>7} "
            f"{r['balanced_acc']:>8} {r['f1_weighted']:>7} {r['f1_macro']:>7} "
            f"{r['rem_recall']:>8} {r['transition_acc']:>7} {r['worst_animal_kappa']:>8}"
        )
    print("\nPer-animal kappa detail:")
    for r in rows:
        print(f"  {r['feature_set']}: {json.dumps(r['per_animal_kappa'])}")
    return rows


def main():
    p = argparse.ArgumentParser(description="LOO out-of-sample feature-set comparison")
    p.add_argument("--v1", help="Path to v1 feature-store CSV")
    p.add_argument("--v2", help="Path to v2 feature-store CSV")
    args = p.parse_args()

    stores = {}
    if args.v1:
        stores["v1"] = args.v1
    if args.v2:
        stores["v2"] = args.v2
    if not stores:
        p.error("provide at least one of --v1 / --v2")
    run(stores)


if __name__ == "__main__":
    main()
