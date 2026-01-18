from typing import Optional, Union

from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import time
import pandas as pd

from model.cross_validation import get_cv_splitter


def find_best_params(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    search_space: dict[str, list],
    random_state: int = 42,
    eval_metric: str = "f1_weighted",
    cv_folds: int = 5,
    n_iter: int = 10,
    cv_splitter: Optional[Union[BaseCrossValidator, str]] = None,
    groups: Optional[pd.Series] = None,
) -> tuple:
    """Find best hyperparameters for XGBoost Classifier using RandomizedSearchCV.

    Args:
        X_train (pd.DataFrame): Dataframe containing the values of the predictor variables.
        y_train (pd.DataFrame): Dataframe containing the values of the target variable.
        search_space (dict[str, list]): Dictionary containing the parameter names and values to be evaluated.
        random_state (int): Random state for repeatability. Default value is 42.
        eval_metric (str): The scoring metric used by RandomizedSearchCV. Default value is "f1_weighted"
        cv_folds (int): Number of folds for cross-validation. Default value is 5. Ignored if cv_splitter provided.
        n_iter (int): Number of parameter values that are sampled. Default value is 10.
        cv_splitter (BaseCrossValidator or str, optional): Custom CV splitter for time-series aware validation.
            Can be a sklearn CV object or one of: "group_time_series", "leave_one_out".
            If None, uses standard k-fold (not recommended for time-series data).
        groups (pd.Series, optional): Group labels (e.g., ID_day) for group-aware CV.
            Required if cv_splitter is group-aware.

    Returns:
        (dict[str, list]): The dictionary containing the optimal hyperparameter values for the model.
        (float): The duration of time it took to complete the tuning process.

    Examples:
        # Using time-series aware CV (recommended):
        >>> best_params, duration = find_best_params(
        ...     X_train, y_train, search_space,
        ...     cv_splitter="group_time_series",
        ...     groups=train_df['ID_day']
        ... )

        # Using standard k-fold (legacy behavior):
        >>> best_params, duration = find_best_params(X_train, y_train, search_space)
    """
    # Label encode y_train
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Define the XGBoost Classifier
    xgb_classifier = XGBClassifier(
        tree_method="hist"
    )  # tree_method='hist' significantly speeds up training

    # Set up cross-validation
    if cv_splitter is None:
        # Legacy behavior: standard k-fold (not recommended for time-series)
        cv = cv_folds
    elif isinstance(cv_splitter, str):
        # Get CV splitter by name
        cv = get_cv_splitter(strategy=cv_splitter, n_splits=cv_folds)
    else:
        # Use provided CV object
        cv = cv_splitter

    # Perform Randomized Search Cross Validation to find the best hyperparameters
    search_start = time.time()
    random_search = RandomizedSearchCV(
        estimator=xgb_classifier,
        param_distributions=search_space,
        verbose=2,  # Prints model training info to console
        n_iter=n_iter,
        cv=cv,
        scoring=eval_metric,
        random_state=random_state,
    )

    # Fit with groups if provided (needed for group-aware CV)
    if groups is not None:
        random_search.fit(X_train, y_train_encoded, groups=groups)
    else:
        random_search.fit(X_train, y_train_encoded)

    search_end = time.time()
    search_duration = search_end - search_start

    # Get the best hyperparameters
    best_params = random_search.best_params_

    return best_params, search_duration


def train_model(X: pd.DataFrame, y: pd.DataFrame, params: dict) -> tuple:
    """

    Args:
        X (pd.DataFrame): Dataframe containing the predictor variables
        y (pd.DataFrame): Dataframe containing the target variable
        params (dict[str,list]): The hyperparameter values for the model.

    Returns:
        (XGBoostClassifier): The trained XGBoost model.
        (pd.Series): The decoded predicted values of y.
        (float): The time it took to fit the model.
        (np.array): Label encodings from sklearn.LabelEncoder().
    """

    # Encode y
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train the model
    train_start = time.time()
    model = XGBClassifier(**params, tree_method="hist")
    model.fit(X, y_encoded)
    train_end = time.time()
    time_to_fit = train_end - train_start

    # Make predictions on the test set
    y_pred = model.predict(X)

    # Decode the predicted values
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    return model, y_pred_decoded, time_to_fit, label_encoder
