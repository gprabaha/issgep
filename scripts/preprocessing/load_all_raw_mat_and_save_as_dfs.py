# scripts/preprocessing/load_all_raw_mat_and_save_as_dfs.py

"""
This script loads all raw behavioral data (e.g., eye positions, pupil sizes, ROI bounding box vertices, etc.)
from .mat files for each session and run listed in the configuration, processes them into pandas DataFrames,
and saves the concatenated DataFrames for each data type as .pkl files for downstream analysis.

The configuration specifies:
- which sessions and runs to include,
- which types of behavioral data to process.
"""

import os
import logging
from collections import defaultdict
import pandas as pd

from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.config_utils import ensure_config_exists
from socialgaze.data.extract_data_from_mat_files import generate_behav_data_loader_dict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_all_data(config: BaseConfig, behav_data_loader_registry: dict) -> dict:
    """
    Loads and processes behavioral data for all specified sessions and runs.

    Parameters
    ----------
    config : BaseConfig
        Configuration object containing session/run lists and registry of how to load/process each data type.

    Returns
    -------
    dict
        A dictionary mapping data type (e.g., 'positions', 'neural_timeline') to a list of pandas DataFrames.
    """
    data_collectors = defaultdict(list)
    for session_name in config.session_names:
        for run_number in config.runs_by_session.get(session_name, []):
            run_number = str(run_number)
            logger.info(f"Getting data for: session: {session_name}, run: {run_number}")
            for data_type, registry_entry in behav_data_loader_registry.items():
                path_func = registry_entry["path_func"]       # Function to get file path
                process_func = registry_entry["process_func"] # Function to load/process data
                agent_specific = registry_entry["agent_specific"]
                if agent_specific:
                    # Separate data loading for each monkey
                    for agent in ["m1", "m2"]:
                        path = path_func(config, session_name, run_number)
                        df = process_func(path, session_name, run_number, agent)
                        if df is not None:
                            data_collectors[data_type].append(df)
                else:
                    # Shared data loading (no agent distinction)
                    path = path_func(config, session_name, run_number)
                    df = process_func(path, session_name, run_number)
                    if df is not None:
                        data_collectors[data_type].append(df)
    return data_collectors


def save_loaded_data_as_dataframes(data_collectors: dict, output_dir):
    """
    Combines and saves all collected data as pickle (.pkl) files by data type.

    Parameters
    ----------
    data_collectors : dict
        Dictionary where each key is a data type and the value is a list of pandas DataFrames.

    output_dir : str or Path
        Directory where the output .pkl files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Combining all dataframes and saving")
    for data_type, dfs in data_collectors.items():
        if dfs:
            out_path = output_dir / f"{data_type}.pkl"
            pd.concat(dfs).to_pickle(out_path)
            logger.info(f"Saved {data_type}.pkl to {out_path}")
    logger.info("All raw data loaded and saved as DataFrames.")


def main():
    """
    Main function to load configuration, collect all data, and save results.
    """
    config_path = "src/socialgaze/config/saved_configs/milgram_default.json"
    ensure_config_exists(config_path)
    config = BaseConfig(config_path=config_path)

    behav_data_loader_registry = generate_behav_data_loader_dict(config.behav_data_types)
    data_collectors = collect_all_data(config, behav_data_loader_registry)
    save_loaded_data_as_dataframes(data_collectors, config.processed_data_dir)


if __name__ == "__main__":
    main()
