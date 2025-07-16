# scripts/behav_analysis/00_preprocessing.py

"""
Preprocesses all raw behavioral data by:
1. Loading .mat files for each session/run listed in the config.
2. Converting them into DataFrames for positions, pupil sizes, ROI vertices, and neural timeline.
3. Pruning and interpolating missing data.
4. Saving the cleaned DataFrames to .pkl files for downstream analysis.
"""

import pdb
import logging
from socialgaze.config.base_config import BaseConfig
from socialgaze.data.gaze_data import GazeData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config = BaseConfig()
    gaze_data = GazeData(config)

    pdb.set_trace()

    logger.info("Step 1: Loading raw .mat data...")
    gaze_data.load_raw_data_from_mat_files()

    logger.info("Step 2: Pruning and interpolating...")
    gaze_data.prune_and_interpolate()

    logger.info("Step 3: Storing the length of each run...")
    gaze_data.get_run_lengths()

    logger.info("Step 4: Saving cleaned DataFrames...")
    gaze_data.save_as_dataframes()

    logger.info("Done.")
