# src/socialgaze/data/spike_data.py

import os
import logging
from typing import Optional
import pandas as pd
import numpy as np

from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.loading_utils import load_df_from_pkl, load_mat_from_path
from socialgaze.utils.saving_utils import save_df_to_pkl
from socialgaze.utils.path_utils import get_spike_times_mat_path, get_spike_df_pkl_path

logger = logging.getLogger(__name__)


class SpikeData:
    """
    SpikeData handles the loading, saving, and access of spike time data.

    Features:
    - Loads spike times from a .mat file into a pandas DataFrame.
    - Allows saving and loading of spike data as .pkl for faster access.
    - Provides `get_data()` to retrieve the dataframe with safety checks.
    """


    def __init__(self, config: BaseConfig):
        """
        Initializes the SpikeData loader.

        Args:
            config (BaseConfig): Configuration object containing path settings.
        """
        self.config = config
        self.spike_df: Optional[pd.DataFrame] = None


    def load_from_mat(self):
        """
        Loads spike times from the raw .mat file and converts them to a DataFrame.
        """
        spike_path = self.config.spiketimes_mat_path
        logger.info(f"Loading spike times from {spike_path}")
        mat_data = load_mat_from_path(spike_path)
        spike_data = mat_data['unit_spiketimes'][0][0]
        spike_dict = {}
        for key in spike_data.dtype.names:
            spike_dict[key] = _flatten_nested_arrays(spike_data[key].squeeze())
        df = pd.DataFrame(spike_dict)
        df.rename(columns={'session': 'session_name'}, inplace=True)
        self.spike_df = df
        logger.info(f"Loaded spike data with {len(df)} rows.")


    def save_to_pkl(self):
        """
        Saves the spike DataFrame to a .pkl file.
        """
        if self.spike_df is None:
            logger.warning("No spike data to save.")
            return
        save_path = self.config.spiketimes_df_path
        logger.info(f"Saving spike data to {save_path}")
        save_df_to_pkl(self.spike_df, save_path)


    def load_from_pkl(self):
        """
        Loads the spike DataFrame from a previously saved .pkl file.
        """
        load_path = self.config.spiketimes_df_path
        logger.info(f"Loading spike data from {load_path}")
        self.spike_df = load_df_from_pkl(load_path)


    def get_data(self, session_name: Optional[str] = None, unit_uuid: Optional[str] = None) -> pd.DataFrame:
        """
        Returns the spike data, optionally filtered by session name and/or unit UUID.

        Loads from saved .pkl if not already in memory.

        Args:
            session_name (str, optional): Filter for a specific session.
            unit_uuid (str, optional): Filter for a specific unit UUID.

        Raises:
            RuntimeError: If spike data cannot be loaded or is missing.

        Returns:
            pd.DataFrame: Filtered spike data.
        """
        if self.spike_df is None:
            try:
                self.load_from_pkl()
            except Exception as e:
                logger.error(f"Failed to load spike data from .pkl: {e}")
                raise RuntimeError("Spike data not loaded and no saved pickle available.")

        df = self.spike_df

        if session_name is not None:
            df = df[df["session_name"] == session_name]
        if unit_uuid is not None and "unit_uuid" in df.columns:
            df = df[df["unit_uuid"] == unit_uuid]

        return df.reset_index(drop=True)



def _flatten_nested_arrays(arr):
    """
    Flatten nested arrays inside the given array, handling multi-element arrays.
    """
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [
            elem.item() if isinstance(elem, np.ndarray) and elem.size == 1 else
            elem[0] if isinstance(elem, np.ndarray) and elem.size > 1 and isinstance(elem[0], np.ndarray) else elem
            for elem in arr
        ]
    return arr
