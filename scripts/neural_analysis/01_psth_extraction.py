# scripts/analysis/01_psth_extraction.py

"""
Script: 01_psth_extraction.py

Description:
    This script initializes all necessary data/config objects and runs per-fixation PSTH extraction
    using gaze and spike data. Fixations are labeled as interactive or not using mutual fixation density.
    The script computes:
      - Per-fixation PSTHs for each unit
      - Mean PSTH per unit and fixation category
      - Mean PSTH per unit, fixation category, and interactivity status
    All resulting DataFrames are saved to disk.

Run:
    python scripts/analysis/01_psth_extraction.py

"""

import logging

import pdb

from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.psth_config import PSTHConfig
from socialgaze.config.interactivity_config import InteractivityConfig

from socialgaze.data.gaze_data import GazeData
from socialgaze.data.spike_data import SpikeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.features.psth_extractor import PSTHExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing config objects...")
    base_config = BaseConfig()
    fixation_config = FixationConfig()
    neural_config = PSTHConfig()
    interactivity_config = InteractivityConfig()

    logger.info("Initializing data managers...")
    gaze_data = GazeData(config=base_config)
    spike_data = SpikeData(config=base_config)

    logger.info("Initializing detectors...")
    fixation_detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    interactivity_detector = InteractivityDetector(config=interactivity_config)

    logger.info("Creating PSTH extractor...")
    extractor = PSTHExtractor(
        config=neural_config,
        gaze_data=gaze_data,
        spike_data=spike_data,
        fixation_detector=fixation_detector,
        interactivity_detector=interactivity_detector,
    )

    pdb.set_trace()

    try:
        logger.info("Starting per-fixation PSTH extraction...")
        extractor.compute_psth_per_trial(overwrite=False)

        logger.info("Computing mean PSTH per category...")
        extractor.compute_avg_psth_per_category()

        logger.info("Computing mean PSTH per category and interactivity...")
        extractor.compute_avg_psth_per_category_and_interactivity()

        logger.info("Saving all PSTH dataframes to disk...")
        extractor.save_dataframes()

        logger.info("PSTH extraction and save completed successfully.")
        
        # Display heads of the resulting dataframes
        print("\n--- Head of psth_per_trial ---")
        print(extractor.psth_per_trial.head())
        
        print("\n--- Head of avg_psth_per_category ---")
        print(extractor.avg_psth_per_category.head())
        
        print("\n--- Head of avg_psth_per_category_and_interactivity ---")
        print(extractor.avg_psth_per_category_and_interactivity.head())

    except Exception as e:
        logger.exception(f"PSTH extraction failed: {e}")


if __name__ == "__main__":
    main()
