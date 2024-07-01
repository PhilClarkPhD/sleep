import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings


def make_save_dir(artifacts_path: os.path, model_id: str) -> os.path:
    """
    Check for a directory to save model artifacts. If not found, make one
    Args:
        artifacts_path (os.path): parent directory
        model_id (os.path): model id for current model. will be the name of sub directory containing this model's
        artifacts.

    Returns:
        save_dir(os.path): directory in which current model's artifacts will be saved

    """
    save_dir = os.path.join(artifacts_path, model_id)
    if not os.path.isdir(save_dir):
        warnings.warn(f"{save_dir} does not exist. Creating it now...", UserWarning)
        os.makedirs(save_dir, exist_ok=True)

    return save_dir


def save_model_artifacts(
    save_dir: str,
    model: XGBClassifier,
    metadata: dict,
    encoder: LabelEncoder,
    file_name: str,
) -> None:
    """Saves the model and metadata in a .pkl file.

    Args:
        save_dir (str): Path to the directory where you want the files saved.
        model (XGBClassifier): The fit model
        metadata (dict[str,list]): Dictionary containing the metadata for the model.
        encoder (LabelEncoder): sklearn LabelEncoder used when training the model.
        file_name (str): The name of the file that will be saved.

    Returns:
        .pkl file containing model, metadata, and encoder in a tuple

    """

    # Check if path is valid
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"The path {save_dir} does not exist")

    else:
        joblib.dump(
            (model, metadata, encoder), f"{save_dir}/{file_name}.pkl"
        )  # pkl and save the model/hyperparams


def save_test_data(save_dir: os.path, test_data: pd.DataFrame, model_id: str) -> None:
    """

    Args:
        save_dir (os.path): directory for saving model artifacts
        test_data (pd.DataFrame): df w/ test data, including predicted scores
        model_id (str): model id

    Returns:
        None
    """
    save_file_path = os.path.join(save_dir, f"{model_id}_test_data.csv")
    test_data.to_csv(save_file_path)
