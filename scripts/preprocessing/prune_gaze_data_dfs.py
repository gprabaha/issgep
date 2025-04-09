# scripts/preprocessing/prune_gaze_data_dfs.py


from pathlib import Path
import logging
import pandas as pd

from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.config_utils import ensure_config_exists

import pdb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config_path = "src/socialgaze/config/saved_configs/milgram_default.json"
    ensure_config_exists(config_path)

    config = BaseConfig(config_path=config_path)

    behav_data_types = config.behav_data_types
    dataframe_dict = {}
    for data_type in behav_data_types:
        file_path = processed_data_dir / f"{data_type}.pkl"
        dataframe_dict.setdefault(data_type, pd.DataFrame())
        dataframe_dict[data_type] = load_df_from_pkl(file_path)
    
    # remove starting and trailing NaNs from M1 and M2's position data and remove the corresponding indices from pupil and timeline
    
    # use non-linear interpolation to fill in cases with sparse NaNs in the data


if __name__ == "__main__":
    main()