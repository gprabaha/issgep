# scripts/preprocessing/load_all_raw_mat_and_save_as_dfs.py

"""
This script loads all raw behavioral data (e.g., eye positions, pupil sizes, ROI bounding box vertices, etc.)
from .mat files for each session and run listed in the configuration, processes them into pandas DataFrames,
and saves the concatenated DataFrames for each data type as .pkl files for downstream analysis.

The configuration specifies:
- which sessions and runs to include,
- which types of behavioral data to process.
"""

from socialgaze.config.base_config import BaseConfig
from socialgaze.analysis.gaze_data import GazeData


def main():
    config = BaseConfig()
    gaze_data = GazeData(config)

    # Load everything
    gaze_data.load_raw_data_from_mat_files()

    # Save everything
    gaze_data.save_as_dataframes(config.processed_data_dir)
