"""
Model training pipeline.

Workflow:
    1. Load feature store data
    2. Split into train/test (time-series aware holdout)
    3. Tune hyperparameters on train set using TSCV
    4. Evaluate best params on held-out test set
    5. Train final model on ALL data with best params
    6. Save model artifacts + test set predictions

Usage:
    python -m model.train
    python -m model.train --config model/model_config.json
    python -m model.train --config model/model_config.json --skip-tuning
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from model.config import FEATURE_COLS
from model.cross_validation import get_cv_splitter
from model.diagnostics import evaluate_model
from model.save_model import make_save_dir, save_model_artifacts, save_test_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.ml.sleep_functions import apply_rule_based_filter


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def split_train_test(
    df: pd.DataFrame,
    train_size: float,
    group_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-series aware train/test split, respecting group boundaries."""
    df = df.sort_values(by=[group_col, time_col])
    train_parts, test_parts = [], []

    for group in df[group_col].unique():
        group_df = df.loc[df[group_col] == group]
        n_train = int(len(group_df) * train_size)
        train_parts.append(group_df.iloc[:n_train])
        test_parts.append(group_df.iloc[n_train:])

    return pd.concat(train_parts), pd.concat(test_parts)


def tune_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    search_space: dict,
    cv_folds: int = 5,
    n_iter: int = 50,
    random_state: int = 42,
) -> tuple[dict, float]:
    """Find best hyperparameters using TSCV on train set only."""
    from sklearn.model_selection import RandomizedSearchCV

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    cv = get_cv_splitter(
        strategy="group_time_series",
        n_splits=cv_folds,
        gap=6,
    )

    xgb = XGBClassifier(tree_method="hist")

    log.info(f"Running RandomizedSearchCV: {n_iter} iterations, {cv_folds} folds")
    t0 = time.time()

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_weighted",
        random_state=random_state,
        verbose=1,
    )
    search.fit(X_train, y_encoded, groups=groups)

    duration = time.time() - t0
    log.info(f"Best params found in {duration:.1f}s: {search.best_params_}")
    log.info(f"Best CV score: {search.best_score_:.4f}")

    return search.best_params_, duration


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
) -> tuple[XGBClassifier, LabelEncoder, float]:
    """Train final model on ALL data with best params."""
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = XGBClassifier(**params, tree_method="hist")

    t0 = time.time()
    model.fit(X, y_encoded)
    duration = time.time() - t0

    log.info(f"Final model trained on {len(X)} epochs in {duration:.1f}s")
    return model, label_encoder, duration


def run(config_path: str, skip_tuning: bool = False):
    config = load_config(config_path)

    # Config values
    model_name = config["model_name"]
    model_version = config["model_version"]
    model_id = f"{model_name}_{model_version}"
    feature_cols = config.get("feature_cols", FEATURE_COLS)
    target_col = config["target_col"]
    group_col = config["group_col"]
    time_col = config["time_series_index"]
    use_filter = config.get("use_rule_based_filter", True)

    # Load data
    feature_path = os.path.join(
        config["paths"]["feature_store_directory"],
        config["paths"]["feature_store_table"],
    )
    log.info(f"Loading features from {feature_path}")
    df = pd.read_csv(feature_path)

    if config.get("drop_unscored", True):
        n_before = len(df)
        df = df.loc[df[target_col] != "Unscored"]
        log.info(f"Dropped {n_before - len(df)} unscored epochs, {len(df)} remaining")

    # --- Step 1: Train/test split ---
    train_df, test_df = split_train_test(
        df, config["train_size"], group_col, time_col
    )
    log.info(f"Split: {len(train_df)} train, {len(test_df)} test")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # --- Step 2: Hyperparameter tuning (TSCV on train set only) ---
    if skip_tuning and "best_params" in config:
        best_params = config["best_params"]
        search_duration = 0.0
        log.info(f"Skipping tuning, using params from config: {best_params}")
    else:
        best_params, search_duration = tune_hyperparams(
            X_train=X_train,
            y_train=y_train,
            groups=train_df[group_col],
            search_space=config["search_space"],
            cv_folds=config.get("cv_folds", 5),
            n_iter=config.get("n_iter", 50),
            random_state=config.get("random_state", 42),
        )

    # --- Step 3: Evaluate on held-out test set ---
    log.info("Evaluating on held-out test set...")
    eval_encoder = LabelEncoder()
    eval_encoder.fit(df[target_col])

    eval_model = XGBClassifier(**best_params, tree_method="hist")
    eval_model.fit(X_train, eval_encoder.transform(y_train))

    y_test_pred_encoded = eval_model.predict(X_test)
    y_test_pred = eval_encoder.inverse_transform(y_test_pred_encoded)

    if use_filter:
        y_test_pred = apply_rule_based_filter(y_test_pred)

    metrics = evaluate_model(y_test.values, y_test_pred, analyze_transitions=True)
    print("\n" + str(metrics))

    test_score = f1_score(y_test, y_test_pred, average="weighted")
    log.info(f"Held-out test F1 (weighted): {test_score:.4f}")

    # --- Step 4: Train final model on ALL data ---
    log.info("Training final model on all data...")
    X_all = df[feature_cols]
    y_all = df[target_col]

    final_model, label_encoder, fit_duration = train_final_model(
        X_all, y_all, best_params
    )

    # --- Step 5: Save artifacts ---
    artifacts_path = config["paths"]["artifacts_path"]
    save_dir = make_save_dir(artifacts_path, model_id)

    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "model_id": model_id,
        "model_artifacts_path": save_dir,
        "timestamp": str(datetime.datetime.now()),
        "feature_store_table": config["paths"]["feature_store_table"],
        "feature_cols": feature_cols,
        "target_col": target_col,
        "group_col": group_col,
        "train_size": config["train_size"],
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "n_total_rows": len(X_all),
        "search_space": config["search_space"],
        "best_params": best_params,
        "search_duration": search_duration,
        "fit_duration": fit_duration,
        "use_rule_based_filter": use_filter,
        # Held-out test metrics
        "test_f1_weighted": test_score,
        "test_cohen_kappa": metrics.cohen_kappa,
        "test_balanced_accuracy": metrics.balanced_accuracy,
        "test_class_metrics": metrics.class_metrics,
        "test_transition_accuracy": metrics.transition_accuracy,
        "label_encoder": label_encoder.classes_.tolist(),
        "feature_importances": dict(zip(feature_cols, final_model.feature_importances_.tolist())),
        "notes": config.get("notes", ""),
    }

    save_model_artifacts(save_dir, final_model, metadata, label_encoder, model_id)
    log.info(f"Model saved to {save_dir}/{model_id}.pkl")

    # Save test set predictions for later analysis
    df_test_out = test_df[feature_cols].copy()
    df_test_out["score"] = y_test.values
    df_test_out["predicted_score"] = y_test_pred
    save_test_data(save_dir, df_test_out, model_id)
    log.info(f"Test predictions saved to {save_dir}/{model_id}_test_data.csv")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train sleep scoring model")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "model_config.json"),
        help="Path to model config JSON",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter tuning, use best_params from config",
    )
    args = parser.parse_args()

    run(args.config, args.skip_tuning)


if __name__ == "__main__":
    main()
