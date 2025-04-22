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