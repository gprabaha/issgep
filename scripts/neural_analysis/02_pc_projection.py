# scripts/neural_analysis/02_pc_projection.py

"""
Script: 02_pc_projection.py

Description:
    This script initializes all necessary data/config objects and runs PCA on 
    population-averaged firing rates grouped by fixation category or interactivity.
    The resulting PCA fits and projections are saved to disk for downstream analyses.
    Then, it computes distances and angles between projected trajectories for each transform.

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
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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

    allowed_fits = {"fit_avg_face_obj", "fit_int_non_int_face_obj"}
    allowed_transforms = {"transform_avg_face_obj", "transform_int_non_int_face_obj"}
    
    try:
        logger.info("Creating PC projector...")
        pc_projector = PCProjector(config=pca_config, psth_extractor=psth_extractor)

        # Fit and project
        if pca_config.use_parallel:
            logger.info("Running PCA fits in parallel...")
            run_joblib_parallel(
                delayed(pc_projector.fit)(fit_spec)
                for fit_spec in FIT_SPECS
                if fit_spec.name in allowed_fits
            )

            logger.info("Running PCA projections in parallel...")
            run_joblib_parallel(
                delayed(pc_projector.project)(fit_spec.name, transform_spec)
                for fit_spec, transform_spec in product(FIT_SPECS, TRANSFORM_SPECS)
                if fit_spec.name in allowed_fits and transform_spec.name in allowed_transforms
            )
        else:
            for fit_spec in FIT_SPECS:
                if fit_spec.name not in allowed_fits:
                    continue
                logger.info(f"Running PCA fit: {fit_spec.name}")
                pc_projector.fit(fit_spec)

            for fit_spec, transform_spec in product(FIT_SPECS, TRANSFORM_SPECS):
                if fit_spec.name not in allowed_fits or transform_spec.name not in allowed_transforms:
                    continue
                logger.info(f"Running PCA projection: fit={fit_spec.name} | transform={transform_spec.name}")
                pc_projector.project(fit_spec_name=fit_spec.name, transform_spec=transform_spec)

        logger.info("PCA projection script completed successfully.")

        # # Compare trajectories
        # logger.info("\n--- Comparing category trajectories ---")

        # # Define allowed spec names
        # allowed_fits = {"fit_avg_face_obj", "fit_int_non_int_face_obj"}
        # allowed_transforms = {"transform_avg_face_obj", "transform_int_non_int_face_obj"}

        # for fit_spec, transform_spec in product(FIT_SPECS, TRANSFORM_SPECS):
        #     if (
        #         fit_spec.name not in allowed_fits
        #         or transform_spec.name not in allowed_transforms
        #     ):
        #         continue

        #     key = f"{fit_spec.name}__{transform_spec.name}"
        #     available_regions = pc_projector.get_available_fit_transform_region_keys().get(key, [])

        #     for region in available_regions:
        #         logger.info(f"\nComparing trajectories for: fit={fit_spec.name}, transform={transform_spec.name}, region={region}")
        #         try:
        #             results = pc_projector.compare_category_trajectories(fit_spec.name, transform_spec.name, region)
        #             for res in results:
        #                 print(
        #                     f"{res['category_1']} vs {res['category_2']} | "
        #                     f"Euclidean Distance: {res['euclidean_distance']:.4f} | "
        #                     f"Angle: {res['vector_angle_deg']:.2f}° | "
        #                     f"Trajectory Length diff: {res['trajectory_length_diff']:.4f} | "
        #                     f"Procrustes Disparity: {res['procrustes_disparity']:.4f}"
        #                 )
        #         except Exception as e:
        #             logger.warning(f"Failed to compare trajectories for {key} in region {region}: {e}")

    except Exception as e:
        logger.exception(f"PCA projection failed: {e}")


if __name__ == "__main__":
    main()
