import os
import joblib
import numpy as np


def save_model_and_params(
    save_dir: str,
    model,
    metadata: dict,
    model_name: str,
) -> None:
    """Saves the model and metadata in a .pkl file.

    Args:
        save_dir (str): Path to the directory where you want the files saved.
        model (XGBoost model): The fit model
        metadata (dict[str,list]): Dictionary containing the metadata for the model.
        model_name (str): The name of the model, should be same as metadata['model_name'].

    Returns:
        None
    """

    # Check if path is valid
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"The path {save_dir} does not exist")

    else:
        joblib.dump(
            (model, metadata), f"{save_dir}/{model_name}.pkl"
        )  # pkl and save the model/hyperparams


def save_encoder(
    save_dir: str,
    label_encodings: np.array,
    model_name: str,
) -> None:
    """Saves the label encodings in a .pkl file.

    Args:
        save_dir (str): The path for the directory where you want to save the file.
        label_encodings (np.array): Label encodings. Should be the result of calling encoder.classes_
        model_name (str): Name of the model for which these are the encodings.

    Returns:
        None
    """

    # Check if path is valid
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"The path {save_dir} does not exist")

    else:
        joblib.dump(label_encodings, f"{save_dir}/{model_name}_label_encodings.pkl")
