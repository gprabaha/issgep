# scripts/neural_analysis/02_pc_projection.py

"""
Script: 02_pc_projection.py

Description:
    This script initializes all necessary data/config objects and runs PCA on 
    population-averaged firing rates grouped by fixation category or interactivity.
    The resulting PCA fits and projections are saved to disk for downstream analyses.

Run:
    python scripts/neural_analysis/02_pc_projection.py
"""

import logging
from itertools import product
from joblib import delayed

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

from socialgaze.specs.pca_specs import FIT_SPECS, TRANSFORM_SPECS
from socialgaze.utils.parallel_utils import run_joblib_parallel

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
        logger.info("Creating PC projector...")
        pc_projector = PCProjector(config=pca_config, psth_extractor=psth_extractor)

        if pca_config.use_parallel:
            logger.info("Running PCA fits in parallel...")
            run_joblib_parallel(
                jobs=[delayed(pc_projector.fit)(fit_spec) for fit_spec in FIT_SPECS],
                n_jobs=pca_config.num_cpus,
            )

            logger.info("Running PCA projections in parallel...")
            run_joblib_parallel(
                jobs=[
                    delayed(pc_projector.project)(fit_spec.name, transform_spec)
                    for fit_spec, transform_spec in product(FIT_SPECS, TRANSFORM_SPECS)
                ],
                n_jobs=pca_config.num_cpus,
            )
        else:
            for fit_spec in FIT_SPECS:
                logger.info(f"Running PCA fit: {fit_spec.name}")
                pc_projector.fit(fit_spec)

            for fit_spec, transform_spec in product(FIT_SPECS, TRANSFORM_SPECS):
                logger.info(f"Running PCA projection: fit={fit_spec.name} | transform={transform_spec.name}")
                pc_projector.project(fit_spec.name, transform_spec)

        logger.info("PCA projection script completed successfully.")

    except Exception as e:
        logger.exception(f"PCA projection failed: {e}")


if __name__ == "__main__":
    main()
