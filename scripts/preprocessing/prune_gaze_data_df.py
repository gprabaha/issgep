# scripts/preprocessing/prune_gaze_data_df.py


from pathlib import Path
import pandas as pd

from socialgaze.config.base_config import BaseConfig
from socialgaze.util.loading_utils import load_df_from_pkl
from socialgaze.utils.config_utils import ensure_config_exists

import pdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    config_path = "src/socialgaze/config/saved_configs/milgram_default.json"
    ensure_config_exists(config_path)

    config = BaseConfig(config_path=config_path)

    processed_data_dir = Path(config.processed_data_dir)
    ephys_days_and_monkeys_filepath  = processed_data_dir / "ephys_days_and_monkeys.pkl"
    ephys_days_and_monkeys_df = load_df_from_pkl(ephys_days_and_monkeys_filepath)

    behav_data_types = config.behav_data_types
    dataframe_dict = {}
    for data_type in behav_data_types:
        file_path = processed_data_dir / f"{data_type}.pkl"
        dataframe_dict.setdefault(data_type, pd.DataFrame())
        dataframe_dict[data_type] = load_df_from_pkl(file_path)


if __name__ == "__main__":
    main()