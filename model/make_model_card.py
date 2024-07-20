import numpy as np
import pandas as pd
import joblib
from evidently.report import Report
from evidently.metrics import *
from evidently import ColumnMapping
from utils.load_config import load_config

# Load the model config
config_path = "/Users/phil/philclarkphd/sleep/model/model_config.json"
config = load_config(config_path)

ARTIFACT_PATH = (
    "/Users/phil/philclarkphd/sleep/model_artifacts/XGBoost_1.2.0/XGBoost_1.2.0.pkl"
)

TEST_DATA_PATH = "/Users/phil/philclarkphd/sleep/model_artifacts/XGBoost_1.2.0/XGBoost_1.2.0_test_data.csv"

# Load artifacts
model, metadata, encoder = joblib.load(ARTIFACT_PATH)

# Load test data
df_test = pd.read_csv(TEST_DATA_PATH, index_col=0)

# Map feature importances to feature names
feature_importances = {
    col: importance
    for col, importance in zip(
        metadata["feature_cols"], metadata["train_features_importance"]
    )
}

## Populate Model Card Text ##
model_details = f"""
# Model Details

## Description
* **Model ID**: {metadata["model_id"]}
* **Model Version**: {metadata["model_version"]}
* **Model Type**: {metadata["model_name"]} classifier
* **Training Date**: {metadata["timestamp"]}
* **Feature Store Table**: {metadata["feature_store_table"]}
* **Training Set Proportion**: {metadata["train_size"]}
* **HyperParameters**: {metadata["best_params"]}
* **Feature Importances**: {feature_importances}
* **Training Evaluation**: {metadata["eval_metric"]}
* **Training Score**: {metadata["train_score"]}
* **Notes**: {metadata["notes"]}

## Model Architecture
* **Model Tuning**: True
* **Tuning Algorithm**: RandomSearchCV
* **Search Evaluation Metric**: {metadata["eval_metric"]}
* **Search Space**: {metadata["search_space"]}
* **Random State**: {metadata["random_state"]}
* **CV Folds**: {metadata["cv_folds"]}
* **Number of iterations**: {metadata["n_iter"]}
* **Search Time**: {metadata["search_duration"]}
"""

training_dataset = f"""
# Training dataset

* **Training dataset**: Sleep recordings from Sophie subset 1. Testing new rule-based filter.
* **Sub-groups**: There is just 1 recording per rat, taken at baseline.
* **Pre-processing**: From the raw EEG and EMG data, we derive features to capture EEG and EMG variance (standard deviation, signal amplitude, etc.). We also use a fourier series to calculate the relative amounts of delta, theta and theta/delta power in the EEG signal.
* **Feature Columns**: {metadata["feature_cols"]}
* **Target Column**: {metadata["target_col"]}
"""

model_evaluation = f"""
  # Model evaluation

  * **Evaluation dataset**: A subset {(1-metadata["train_size"])*100}% of the dataset.
  * **Metrics**: Weighted f1 to deal with large class imbalance
  * **Class Representation**: There is a significant imbalance in the class sizes - REM is by far the smallest.
"""


model_summary = """
# Model Summary

* Model run w/ Sophie's data using rule based filter.
"""

## Build Model Card ##
column_mapping = ColumnMapping()

if config["use_rule_based_filter"]:
    column_mapping.prediction = "predicted_score_filtered"
else:
    column_mapping_prediction = "predicted_score"

column_mapping.target = "score"
column_mapping.numerical_features = metadata["feature_cols"]
column_mapping.task = "classification"

model_card = Report(
    metrics=[
        Comment(model_details),
        Comment(training_dataset),
        Comment(model_evaluation),
        ClassificationClassBalance(),
        ClassificationQualityMetric(),
        ClassificationConfusionMatrix(),
        ClassificationQualityByClass(),
        Comment(model_summary),
    ]
)

model_card.run(current_data=df_test, reference_data=None, column_mapping=column_mapping)
SAVE_PATH = f"/Users/phil/philclarkphd/sleep/model_cards/{metadata['model_id']}.html"

model_card.save_html(SAVE_PATH)
