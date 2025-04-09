# src/socialgaze/utils/loading_utils.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_df_from_pkl(filepath):
    logger.info(f"Loading pickle file from {filepath}")
    return pd.read_pickle(filepath) if filepath.exists() else None
