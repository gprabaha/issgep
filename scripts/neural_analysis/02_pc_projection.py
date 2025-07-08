# scripts/neural_analysis/02_pc_projection.py

"""
Script: 02_pc_projection.py

Description:
    Runs population PCA projection analysis:
    - Face vs Object
    - Interactive vs Non-interactive Face

    Uses PSTH data to build unit-by-time matrices,
    fits PCAs, tests generalization, saves:
      - Explained Variance Ratio bar plots
      - 3D PC trajectories + Euclidean distance & angle plots

Run:
    python scripts/neural_analysis/02_pc_projection.py
"""

import logging

from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.interactivity_config import InteractivityConfig
from socialgaze.config.psth_config import PSTHConfig
from socialgaze.config.pca_config import PCAConfig

from socialgaze.data.gaze_data import GazeData
from socialgaze.data.spike_data import SpikeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.features.psth_extractor import PSTHExtractor
from socialgaze.features.pc_projector import PCProjector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("== Initializing config objects ==")
    base_config = BaseConfig()
    fixation_config = FixationConfig()
    interactivity_config = InteractivityConfig()
    psth_config = PSTHConfig()
    pca_config = PCAConfig()

    logger.info("== Initializing data managers ==")
    gaze_data = GazeData(config=base_config)
    spike_data = SpikeData(config=base_config)

    logger.info("== Initializing detectors ==")
    fixation_detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    interactivity_detector = InteractivityDetector(config=interactivity_config)

    logger.info("== Creating PSTH extractor ==")
    psth_extractor = PSTHExtractor(
        config=psth_config,
        gaze_data=gaze_data,
        spike_data=spike_data,
        fixation_detector=fixation_detector,
        interactivity_detector=interactivity_detector,
    )

    logger.info("== Creating PCA projector ==")
    projector = PCProjector(
        psth_extractor=psth_extractor,
        pca_config=pca_config,
    )

    try:
        logger.info("== Running PCA: Face vs Object ==")
        projector.run_face_obj_analysis()

        logger.info("== Running PCA: Interactive vs Non-interactive Face ==")
        projector.run_int_nonint_face_analysis()

        logger.info("PCA projection completed successfully!")

    except Exception as e:
        logger.exception(f"PCA projection failed: {e}")

if __name__ == "__main__":
    main()
