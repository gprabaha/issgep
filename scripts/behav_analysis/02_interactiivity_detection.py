# scripts/behav_analysis/02_interactivity_detection.py

import logging
import argparse

from socialgaze.config.fixation_config import FixationConfig
from socialgaze.data.gaze_data import GazeData
from socialgaze.features.fixation_detector import FixationDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():

    # Load configuration and data
    fixation_config = FixationConfig()
    logger.info("Loaded FixationConfig.")

    gaze_data = GazeData(config=fixation_config)
    logger.info("Initialized GazeData.")

    detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    logger.info("Initialized FixationDetector.")

    detector.load_dataframes()
    logger.info("Loaded fixation and saccade dataframes.")

    detector.add_fixation_category_column()
    logger.info("Added fixation category column.")

    detector.generate_fixation_binary_vectors()
    logger.info("Generated fixation binary vectors.")
    
    detector.save_fixation_binary_vectors()
    logger.info("Fixations df head:")
    logger.info(detector.fixations.head())
    detector.save_dataframes()

    
    
if __name__ == "__main__":
    main()