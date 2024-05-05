def save_model_and_params(
        save_dir: str,
        model,
        params: dict,
        file_name: str,
) -> None:
    import os
    import joblib

    # Check if path is valid
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"The path {save_dir} does not exist")

    else:
        joblib.dump((model, params), f'{save_dir}/{file_name}.pkl') #pkl and save the model/hyperparams



def save_encoder(
        save_dir: str,
        label_encoder,
        file_name: str = 'label_encoder',
) -> None:
    import os
    import joblib

    # Check if path is valid
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"The path {save_dir} does not exist")

    else:
        joblib.dump(label_encoder, f'{save_dir}/{file_name}_label_encodings.pkl')