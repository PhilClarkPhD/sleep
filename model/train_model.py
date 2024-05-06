from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import time
import pandas as pd


def find_best_params(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    search_space: dict[str, list],
    random_state: int = 42,
    eval_metric: str = "f1_weighted",
    cv_folds: int = 5,
    n_iter: int = 10,
) -> tuple:
    """This function finds the best set of hyperparameters for XGBoost Classifier using RandomizedSearchCV.

    Args:
        X_train (pd.DataFrame): Dataframe containing the values of the predictor variables.
        y_train (pd.DataFrame): Dataframe containing the values of the target variable.
        search_space (dict[str, list]): Dictionary containing the parameter names and a corresponding list of values to be evluated.
        random_state (int): Random state for repeatability. Default value is 42.
        eval_metric (str): The scoring metric used by RandomizedSearchCV. Default value is "f1_weighted"
        cv_folds (int): Number of folds for cross-validation. Default value is 5.
        n_iter (int): Number of parameter values that are sampled. Default value is 10.

    Returns:
        (dict[str, list]): The dictionary containing the optimal hyperparameter values for the model.
        (float): The duration of time it took to complete the tuning process.

    """
    # Label encode y_train
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Define the XGBoost Classifier
    xgb_classifier = XGBClassifier(
        tree_method="hist"
    )  # tree_method='hist' significantly speeds up training

    # Perform Randomized Search Cross Validation to find the best hyperparameters
    search_start = time.time()
    random_search = RandomizedSearchCV(
        estimator=xgb_classifier,
        param_distributions=search_space,
        verbose=2,
        n_iter=n_iter,
        cv=cv_folds,
        scoring=eval_metric,
        random_state=random_state,
    )
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

    return model, y_pred_decoded, time_to_fit, label_encoder.classes_
