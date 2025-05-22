# scripts/behav_analysis/02_interactivity_detection.py

"""
Script to detect mutual fixation density and interactivity periods between agents,
based on fixation binary vectors. Results are saved as intermediate data files.

This script loads a precomputed binary vector DataFrame, initializes an
InteractivityDetector, computes mutual fixation densities, extracts interactivity
periods based on a density threshold, and saves both outputs.
"""

import logging
from socialgaze.config.interactivity_config import InteractivityConfig
from socialgaze.features.interactivity_detector import InteractivityDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run interactivity detection pipeline.

    Steps:
        1. Load configuration and binary fixation vectors.
        2. Compute and save mutual fixation density.
        3. Compute and save interactivity periods.
        4. Print previews of both results.
    """
    # Load configuration
    config = InteractivityConfig()
    logger.info("Loaded InteractivityConfig.")

    # Initialize interactivity detector
    detector = InteractivityDetector(config=config)

    # # Run mutual fixation density detection and save results
    # logger.info("Computing mutual fixation density...")
    # detector.detect_mutual_face_fix_density()
    # detector.save_fix_densities()
    # logger.info("Mutual fixation density computed and saved.")

    # # Preview mutual fixation density
    # print("\n=== Head of Mutual Fixation Density DataFrame ===")
    # print(detector.get_density().head())

    # Compute and save interactivity periods
    logger.info("Extracting interactivity periods...")
    detector.compute_interactivity_periods()
    detector.save_interactivity_periods()
    logger.info("Interactivity periods computed and saved.")

    # Preview interactivity periods
    print("\n=== Head of Interactivity Period DataFrame ===")
    print(detector.get_interactivity_periods().head())


if __name__ == "__main__":
    main()
