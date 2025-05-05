# scripts/neural_analysis/02_pc_projection.py

"""
Script: 02_pc_projection.py

Description:
    This script initializes all necessary data/config objects and runs PCA on 
    population-averaged firing rates grouped by fixation category.
    The resulting PCA fits and projections are saved to disk for downstream analyses.

Run:
    python scripts/neural_analysis/02_pc_projection.py
"""

import logging

from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.psth_config import PSTHConfig
from socialgaze.config.interactivity_config import InteractivityConfig
from socialgaze.config.pca_config import PCAConfig

from socialgaze.data.gaze_data import GazeData
from socialgaze.data.spike_data import SpikeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.features.psth_extractor import PSTHExtractor
from socialgaze.features.pc_projector import PCProjector

from socialgaze.specs.pca_specs import PCASPECS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing config objects...")
    base_config = BaseConfig()
    fixation_config = FixationConfig()
    neural_config = PSTHConfig()
    interactivity_config = InteractivityConfig()
    pca_config = PCAConfig()

    logger.info("Initializing data managers...")
    gaze_data = GazeData(config=base_config)
    spike_data = SpikeData(config=base_config)

    logger.info("Initializing detectors...")
    fixation_detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    interactivity_detector = InteractivityDetector(config=interactivity_config)

    logger.info("Creating PSTH extractor...")
    psth_extractor = PSTHExtractor(
        config=neural_config,
        gaze_data=gaze_data,
        spike_data=spike_data,
        fixation_detector=fixation_detector,
        interactivity_detector=interactivity_detector,
    )

    try:
        logger.info("Loading PSTH data...")
        psth_extractor.load_dataframes(which=["by_category"])

        logger.info("Creating PC projection object...")
        pc_projector = PCProjector(config=pca_config, psth_extractor=psth_extractor)

        logger.info("Starting PCA projection on average firing rates...")
        pc_projector.project_avg_firing_rate_by_category()

        logger.info("Saving projections and unit/category orders...")
        pc_projector.save_dataframes()

        logger.info("PCA projection script completed successfully.")

        # Print heads for quick sanity check
        for region, df in pc_projector.pc_projection_dfs.items():
            print(f"\n--- PC projections for region: {region} ---")
            print(df.head())

    except Exception as e:
        logger.exception(f"PC Projection failed: {e}")


if __name__ == "__main__":
    main()
