"""
Script to load, process, and display spike data.

This script demonstrates how to use the SpikeData class to load raw spike data from .mat files,
save them as dataframes, and reload them from disk. The head of the loaded spike dataframe is displayed.
"""

import logging
from socialgaze.config.base_config import BaseConfig
from socialgaze.data.spike_data import SpikeData


import pdb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Initializing configuration and SpikeData handler.")
    config = BaseConfig()
    spike_data = SpikeData(config)

    logger.info("Attempting to load spike data from raw .mat files.")
    spike_data.load_from_mat()

    logger.info("Saving processed spike dataframes to disk.")
    spike_data.save_dataframes()

    logger.info("Reloading spike dataframe from saved files.")
    df = spike_data.get_data()

    if df is not None and not df.empty:
        logger.info("Displaying the head of the loaded spike dataframe:")
        print(df.head())
    else:
        logger.warning("Spike dataframe is empty or not loaded properly.")
