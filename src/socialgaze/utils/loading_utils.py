# src/socialgaze/utils/loading_utils.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_df_from_pkl(filepath):
    """
    Load a DataFrame from a pickle file.

    Args:
        filepath (str): Path to the pickle file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_pickle(filepath) if filepath.exists() else None
