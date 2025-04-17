# src/socialgaze/src/utils/loading_utils.py


from typing import Dict, Any
import json
import logging
import pandas as pd
from scipy.io import loadmat

logger = logging.getLogger(__name__)


def load_config_from_json(config_path: str) -> Dict[str, Any]:
    """
    Loads a config dictionary from a JSON file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        Dict[str, Any]: The config as a dictionary.
    """
    with open(config_path, 'r') as f:
        return json.load(f)



def load_df_from_pkl(filepath: str) -> pd.DataFrame:
    """
    Reads dataframe from a pickle (.pkl) file

    Input:
    ------
    - filepath: str/Path type object containing the path to the dataframe

    Output:
    -------
    - Loaded dataframe or None
    """
    logger.info(f"Loading pickled dataframe from {filepath}")
    return pd.read_pickle(filepath) if filepath.exists() else None

def load_mat_from_path(path) -> dict:
    """
    Loads and parses a .mat file.

    Args:
        path (Path or str): Path to the .mat file.

    Returns:
        dict: Dictionary of parsed MATLAB variables.
    """
    return loadmat(str(path), simplify_cells=False)