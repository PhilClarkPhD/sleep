# Two issues fix in future QA/feature engineering workflow:
# Epochs did not all start at 0 (some started at 1)
# 'Non' and 'Unscored' still present in the 'score' columns for some rats - remove

import datetime
import process_training_data as process_data
import save_model
import train_model
from sklearn.metrics import f1_score

# Set feature path
feature_path = '/Users/phil/philclarkphd/sleep/sleep_data/df_features_041924.csv'

# Load Data
df = process_data.load_data(feature_path)

# Declare inputs for train/test split
feature_cols = ['EEG_std',
                'EEG_ss',
                'EEG_amp',
                'EMG_std',
                'EMG_ss',
                'EMG_events',
                'delta_rel',
                'theta_rel',
                'theta_over_delta']
target_col = 'score'
group_col = 'ID_day'
time_series_idx = 'epoch'
train_size = 0.8

X_train, y_train, X_test, y_test = process_data.train_test_split(df=df, feature_cols=feature_cols,
                                                                 train_size=train_size,
                                                                 time_series_idx=time_series_idx, group_col=group_col,
                                                                 target_col=target_col)

# Find best model params with RandomSearchCV
search_space = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [0, 0.01, 0.1]
}
random_state = 42
cv_folds = 5
n_iter = 10
eval_metric = "f1_weighted"  # Use this on imbalanced multiclass data
best_params, search_duration = train_model.find_best_params(X_train=X_train, y_train=y_train,
                                                            search_space=search_space, random_state=random_state,
                                                            cv_folds=cv_folds, n_iter=n_iter, eval_metric=eval_metric)

# Train model on test data w/ best params
model_0, y_test_pred, *_ = train_model.train_model(X_test, y_test, best_params)

# Evaluate Model
train_feature_importance = model_0.feature_importances_
train_score = f1_score(y_test, y_test_pred, average="weighted")

# Train final model
X = df[feature_cols]
y = df[target_col]
model_version = "1.0"
model_name = f"XGBoost_{model_version}"

final_model, y_pred, time_to_fit, label_encoding = train_model.train_model(X, y, best_params)

model_score = f1_score(y, y_pred, average="weighted")
current_time = datetime.datetime.today()

# Make any notes
notes = '''
Initial model version. Kept in most/all features to maximize predictive accuracy, irrespective of feature
importance.Did not compare different model architectures.
'''

# Populate metadata
n_train_rows = X_train.shape[0]
n_test_rows = X_test.shape[0]

metadata = {
    'model_name': model_name,
    'model_version': model_version,
    'current_path': current_time,
    'feature_path': feature_path,
    'feature_cols': feature_cols,
    'target_col': target_col,
    'group_col': group_col,
    'time_series_idx': time_series_idx,
    'train_size': train_size,
    'n_train_rows': n_train_rows,
    'n_test_rows': n_test_rows,
    'search_space': search_space,
    'random_state': random_state,
    'cv_folds': cv_folds,
    'n_iter': n_iter,
    'eval_metric': eval_metric,
    'best_params': best_params,
    'search_duration': search_duration,
    'train_features_importance': train_feature_importance,
    'train_score': train_score,
    'label_encoding': label_encoding,
    'time_to_fit': time_to_fit,
    'model_score': model_score,
    'notes': notes
}

# Save model and metadata
SAVE_DIR = '/Users/phil/philclarkphd/sleep/model_artefacts'
save_model.save_model_and_params(save_dir=SAVE_DIR, model=final_model, params=best_params, file_name=f"{model_name}")
save_model.save_encoder(save_dir=SAVE_DIR, label_encoder=label_encoding, file_name=f"{model_name}")
