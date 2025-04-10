# src/socialgaze/utils/loading_utils.py

import logging
import pandas as pd

logger = logging.getLogger(__name__)

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
    logger.info(f"Loading pickle file from {filepath}")
    return pd.read_pickle(filepath) if filepath.exists() else None
