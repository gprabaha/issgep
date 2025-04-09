# scripts/preprocessing/load_all_raw_mat_and_save_as_df.py

import os
import sys
import subprocess
import logging
import pandas as pd
from collections import defaultdict

from socialgaze.config.base_config import BaseConfig

import pdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config_path = "src/socialgaze/config/saved_configs/milgram_default.json"

    if not os.path.exists(config_path):
        logger.warning("Config file not found at: %s", config_path)
        logger.info("Attempting to generate config automatically...")
        try:
            subprocess.run(["python", "scripts/setup/make_config_file.py"], check=True)

            logger.info("Config generated at %s", config_path)
        except subprocess.CalledProcessError:
            logger.error("Failed to generate config. Exiting.")
            sys.exit(1)

    config = BaseConfig(config_path=config_path)
    data_collectors = defaultdict(list)  # Stores lists of dfs for each behav_data_type

    for session_name in config.session_names:
        for run_number in config.runs_by_session.get(session_name, []):
            run_number = str(run_number)
            logger.info(f"Getting data for: session: {session_name}, run: {run_number}")

            for data_type in config.behav_data_types:
                registry_entry = config.behav_data_registry[data_type]
                path_func = registry_entry["path_func"]
                process_func = registry_entry["process_func"]
                agent_specific = registry_entry["agent_specific"]

                if agent_specific:
                    for agent in ["m1", "m2"]:
                        path = path_func(session_name, run_number)
                        df = process_func(path, agent, session_name, run_number)
                        if df is not None:
                            data_collectors[data_type].append(df)
                else:
                    path = path_func(session_name, run_number)
                    df = process_func(path, session_name, run_number)
                    if df is not None:
                        data_collectors[data_type].append(df)

    out_dir = config.processed_data_dir
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Combining all dataframes and saving")
    for data_type, dfs in data_collectors.items():
        if dfs:
            out_path = out_dir / f"{data_type}.pkl"
            pd.concat(dfs).to_pickle(out_path)
            logger.info(f"Saved {data_type}.pkl to {out_path}")

    logger.info("All raw data loaded and saved as DataFrames.")

if __name__ == "__main__":
    main()
