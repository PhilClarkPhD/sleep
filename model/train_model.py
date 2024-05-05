from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import time

def find_best_params(
        X_train,
        y_train,
        search_space: dict,
        random_state: int = 42,
        eval_metric: str = "f1_weighted",
        cv_folds: int = 5,
        n_iter: int = 10
) -> tuple:
    """Returns the time it took to find the optimal hyperparameters and the hyperparameters as a dict. Uses
    RandomizedSearchCV and XGBoost Classifier

    Args
    - X_train {}. Training data predictor variables
    - y_train {}. Training data target variable
    -
    """

    # Label encode y_train
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Define the XGBoost Classifier
    xgb_classifier = XGBClassifier(tree_method='hist')  # tree_method='hist' significantly speeds up training

    # Perform Randomized Search Cross Validation to find the best hyperparameters
    search_start = time.time()
    random_search = RandomizedSearchCV(estimator=xgb_classifier, param_distributions=search_space, verbose=2,
                                       n_iter=n_iter, cv=cv_folds, scoring=eval_metric, random_state=random_state)
    random_search.fit(X_train, y_train_encoded)
    search_end = time.time()
    search_duration = search_end - search_start

    # Get the best hyperparameters
    best_params = random_search.best_params_

    return best_params, search_duration


def train_model(
        X,
        y,
        params: dict
) -> tuple:

    # Encode y
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train the model
    train_start = time.time()
    model = XGBClassifier(**params, tree_method='hist')
    model.fit(X, y_encoded)
    train_end = time.time()
    time_to_fit = train_end - train_start

    # Make predictions on the test set
    y_pred = model.predict(X)

    # Decode the predicted values
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    return model, y_pred_decoded, time_to_fit, label_encoder.classes_