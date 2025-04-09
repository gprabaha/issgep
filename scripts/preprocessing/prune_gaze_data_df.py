# scripts/preprocessing/prune_gaze_data_df.py


from pathlib import Path

from socialgaze.config.base_config import BaseConfig
from socialgaze.util.loading_utils import load_df_from_pkl

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

    processed_data_dir = Path(config.processed_data_dir)
    ephys_days_and_monkeys_filepath  = processed_data_dir / "ephys_days_and_monkeys.pkl"
    ephys_days_and_monkeys_df = load_df_from_pkl(ephys_days_and_monkeys_filepath)

    available_timeseries_data = ['positions', 'roi_vertices', 'pupil', 'neural_timeline']



def 


