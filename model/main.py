# TODO:
# Find better way to save the test data and predicted scores (move function into save_model?)
# Two issues to fix in future QA/feature engineering workflow:
#   1. Epochs did not all start at 0 (some started at 1)

import datetime
from train_test_split import train_test_split
import save_model
import train_model
from sklearn.metrics import f1_score
import pandas as pd
import os
from utils.load_config import load_config
from data_processing.sleep_functions import apply_rule_based_filter

# Load the model config
config_path = "/Users/phil/philclarkphd/sleep/model/model_config.json"
config = load_config(config_path)

# model name / version
model_version = config["model_version"]
model_name = config["model_name"]
model_id = model_name + "_" + model_version

##### PATHS #####
artifacts_path = config["paths"]["artifacts_path"]

feature_store_table = config["paths"]["feature_store_table"]
feature_store_directory = config["paths"]["feature_store_directory"]
feature_store_path = os.path.join(feature_store_directory, feature_store_table)

# Make directory for saving model artifacts if it does not exist yet
save_dir = save_model.make_save_dir(artifacts_path, model_id)

# Load Data
df_features = pd.read_csv(feature_store_path)

# Drop unscored epochs
if config["drop_unscored"]:
    df_features = df_features.loc[df_features["score"] != "Unscored"]

# Declare inputs for train/test split
feature_cols = config["feature_cols"]
target_col = config["target_col"]
group_col = config["group_col"]
time_series_index = config["time_series_index"]
train_size = config["train_size"]

train_set, test_set = train_test_split(
    df=df_features,
    train_size=train_size,
    time_series_index=time_series_index,
    group_col=group_col,
)

# Split into train and test
X_train = train_set[feature_cols]
y_train = train_set[target_col]

X_test = test_set[feature_cols]
y_test = test_set[target_col]

# Find best model params with RandomSearchCV
search_space = config["search_space"]
random_state = config["random_state"]
cv_folds = config["cv_folds"]
n_iter = config["n_iter"]
eval_metric = config["eval_metric"]  # Use this on imbalanced multiclass data
best_params, search_duration = train_model.find_best_params(
    X_train=X_train,
    y_train=y_train,
    search_space=search_space,
    random_state=random_state,
    cv_folds=cv_folds,
    n_iter=n_iter,
    eval_metric=eval_metric,
)

# Train model on test data w/ best params
model_0, y_test_pred, *_ = train_model.train_model(X_test, y_test, best_params)

# Evaluate Model
train_feature_importance = model_0.feature_importances_

# Train final model
X = df_features[feature_cols]
y = df_features[target_col]

final_model, y_pred, time_to_fit, label_encoder = train_model.train_model(
    X, y, best_params
)


if config["use_rule_based_filter"]:
    y_pred_filtered = apply_rule_based_filter(y_pred)
    model_score = f1_score(y, y_pred_filtered, average="weighted")

    y_test_pred_filtered = apply_rule_based_filter(y_test_pred)
    train_score = f1_score(y_test, y_test_pred_filtered, average="weighted")
else:
    model_score = f1_score(y, y_pred, average="weighted")
    train_score = f1_score(y_test, y_test_pred, average="weighted")

current_time = datetime.datetime.today()

# Make any notes
notes = config["notes"]

# Populate metadata
n_train_rows = X_train.shape[0]
n_test_rows = X_test.shape[0]

metadata = {
    "model_name": model_name,
    "model_version": model_version,
    "model_id": model_id,
    "model_artifacts_path": save_dir,
    "timestamp": current_time,
    "feature_store_table": feature_store_table,
    "feature_cols": feature_cols,
    "target_col": target_col,
    "group_col": group_col,
    "time_series_index": time_series_index,
    "train_size": train_size,
    "n_train_rows": n_train_rows,
    "n_test_rows": n_test_rows,
    "search_space": search_space,
    "random_state": random_state,
    "cv_folds": cv_folds,
    "n_iter": n_iter,
    "eval_metric": eval_metric,
    "best_params": best_params,
    "search_duration": search_duration,
    "train_features_importance": train_feature_importance,
    "train_score": train_score,
    "label_encoder": label_encoder.classes_,
    "time_to_fit": time_to_fit,
    "model_score": model_score,
    "notes": notes,
}

save_model.save_model_artifacts(
    save_dir=save_dir,
    model=final_model,
    metadata=metadata,
    encoder=label_encoder,
    file_name=model_id,
)

# Make test df
df_test = test_set[feature_cols]
df_test["score"] = y_test
df_test["predicted_score"] = y_test_pred

if config["use_rule_based_filter"]:
    df_test["predicted_score_filtered"] = y_test_pred_filtered


test_data_save_path = save_model.save_test_data(
    save_dir=save_dir, test_data=df_test, model_id=model_id
)
