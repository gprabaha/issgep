# scripts/behav_analysis/02_interactivity_detection.py

import logging
from socialgaze.config.interactivity_config import InteractivityConfig
from socialgaze.utils.loading_utils import load_df_from_pkl
from socialgaze.features.interactivity_detector import InteractivityDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Initialize config
    config = InteractivityConfig()
    logger.info("Loaded InteractivityConfig.")

    # Load binary fixation vectors
    logger.info("Loading fixation binary vector dataframe from %s", config.fix_binary_vec_df_path)
    fix_binary_vector_df = load_df_from_pkl(config.fix_binary_vec_df_path)

    # Initialize and run detector
    detector = InteractivityDetector(fix_binary_vector_df=fix_binary_vector_df, config=config)
    logger.info("Running mutual fixation density analysis...")
    detector.run(True)
    logger.info("Interactivity detection complete.")

    # Show preview
    print("\n=== Head of Mutual Fixation Density DataFrame ===")
    print(detector.get_density().head())


if __name__ == "__main__":
    main()
