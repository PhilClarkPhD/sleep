import os
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


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
        None
    """

    # Check if path is valid
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"The path {save_dir} does not exist")

    else:
        joblib.dump(
            (model, metadata, encoder), f"{save_dir}/{file_name}.pkl"
        )  # pkl and save the model/hyperparams
