"""
Evaluate a saved model on test data.

Loads a saved model artifact and its test data, runs full diagnostics.
Can also evaluate on new data if provided.

Usage:
    python -m model.evaluate --model model_artifacts/XGBoost_1.2.4/XGBoost_1.2.4.pkl
    python -m model.evaluate --model model_artifacts/XGBoost_1.2.4/XGBoost_1.2.4.pkl --data new_data.csv
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from model.config import FEATURE_COLS
from model.diagnostics import evaluate_model, print_confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.ml.sleep_functions import apply_rule_based_filter


def load_model(model_path: str) -> tuple:
    """Load a saved model artifact. Returns (model, metadata, label_encoder)."""
    model, metadata, label_encoder = joblib.load(model_path)
    return model, metadata, label_encoder


def evaluate(
    model_path: str,
    data_path: str = None,
    use_filter: bool = True,
    output_json: str = None,
):
    model, metadata, label_encoder = load_model(model_path)

    model_dir = Path(model_path).parent
    model_id = metadata.get("model_id", Path(model_path).stem)
    feature_cols = metadata.get("feature_cols", FEATURE_COLS)

    # Load test data
    if data_path:
        df = pd.read_csv(data_path)
    else:
        test_data_path = model_dir / f"{model_id}_test_data.csv"
        if not test_data_path.exists():
            print(f"No test data found at {test_data_path}")
            print("Provide --data path to evaluate on specific data")
            sys.exit(1)
        df = pd.read_csv(test_data_path)

    if "score" not in df.columns:
        print("Error: data must have a 'score' column with ground truth labels")
        sys.exit(1)

    X = df[feature_cols]
    y_true = df["score"].values

    # Predict
    y_pred_encoded = model.predict(X)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    if use_filter:
        y_pred = apply_rule_based_filter(y_pred)

    # Full diagnostics
    metrics = evaluate_model(y_true, y_pred, analyze_transitions=True)
    print(str(metrics))
    print()
    print(print_confusion_matrix(y_true, y_pred))

    # Model metadata summary
    print(f"\nModel: {model_id}")
    print(f"Feature importance:")
    importances = metadata.get("feature_importances", {})
    if isinstance(importances, dict):
        for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.4f}")
    elif hasattr(model, "feature_importances_"):
        for feat, imp in sorted(
            zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True
        ):
            print(f"  {feat}: {imp:.4f}")

    if output_json:
        result = metrics.to_dict()
        result["model_id"] = model_id
        result["n_epochs"] = len(y_true)
        result["rule_filter_applied"] = use_filter
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nMetrics saved to {output_json}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved sleep scoring model")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to saved model .pkl file",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to CSV with features + 'score' column. If not provided, uses saved test data.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip rule-based post-processing filter",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Save metrics to JSON file",
    )
    args = parser.parse_args()

    evaluate(args.model, args.data, use_filter=not args.no_filter, output_json=args.output_json)


if __name__ == "__main__":
    main()
