import json
import os


def load_config(config_path: os.path):
    """
    Load the .json config to get model info needed to run main.py"
    Args:
        config_path (os.path): Path to config file. Default is

    Returns:

    """
    with open(config_path, "r") as file:
        config = json.load(file)
    return config
