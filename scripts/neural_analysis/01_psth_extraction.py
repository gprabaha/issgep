# scripts/analysis/01_psth_extraction.py

"""
Script: 01_psth_extraction.py

Description:
    This script initializes all necessary data/config objects and runs per-fixation PSTH extraction
    using gaze and spike data. Fixations are labeled as interactive or not using mutual fixation density.
    The script computes:
      - Per-fixation PSTHs for each unit
      - Mean PSTH per unit: face vs. object
      - Mean PSTH per unit: interactive vs. non-interactive face fixations
    All resulting DataFrames are saved automatically.

Run:
    python scripts/analysis/01_psth_extraction.py
"""

import logging

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
    psth_config = PSTHConfig()
    interactivity_config = InteractivityConfig()

    logger.info("Initializing data managers...")
    gaze_data = GazeData(config=base_config)
    spike_data = SpikeData(config=base_config)

    logger.info("Initializing detectors...")
    fixation_detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    interactivity_detector = InteractivityDetector(config=interactivity_config)

    logger.info("Creating PSTH extractor...")
    extractor = PSTHExtractor(
        config=psth_config,
        gaze_data=gaze_data,
        spike_data=spike_data,
        fixation_detector=fixation_detector,
        interactivity_detector=interactivity_detector,
    )

    try:
        logger.info("Starting per-fixation PSTH extraction...")
        extractor.compute_psth_per_trial()

        logger.info("Computing mean PSTH: face vs. object...")
        extractor.compute_avg_face_obj()

        logger.info("Computing mean PSTH: interactive vs. non-interactive face...")
        extractor.compute_avg_int_non_int_face()

        logger.info("PSTH extraction pipeline completed successfully.")

        # Display heads of the resulting dataframes
        print("\n--- Head of psth_per_trial ---")
        print(extractor.psth_per_trial.head())

        print("\n--- Head of avg_face_obj ---")
        print(extractor.avg_face_obj.head())

        print("\n--- Head of avg_int_non_int_face ---")
        print(extractor.avg_int_non_int_face.head())

    except Exception as e:
        logger.exception(f"PSTH extraction failed: {e}")


if __name__ == "__main__":
    main()
