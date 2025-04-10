# src/socialgaze/utils/saving_utils.py


import os
import json
import pickle
import logging
from typing import Dict, Any
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def save_df_to_pkl(df: pd.DataFrame, path: Path, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """
    Save a pandas DataFrame to a pickle file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - path (Path): The path to save the pickle file to.
    - protocol (int): Pickle protocol version (default: highest).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(df, f, protocol=protocol)
        logger.info(f"Saved DataFrame to: {path}")
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {path}: {e}")
        raise


def save_config_to_json(config_dict: Dict[str, Any], config_path: str) -> None:
    """
    Saves a dictionary to a JSON file.

    Args:
        config_dict (Dict[str, Any]): The config dictionary to save.
        config_path (str): Target path.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        logger.info(f"Saved config to: {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        raise
