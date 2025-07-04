# scripts/visualization/plot_pc_trajectories.py

"""
Script: plot_pc_trajectories.py

Description:
    This script loads PC projections from disk and visualizes the top 3 PC dimensions 
    over time using 3D trajectories. It loops over all fit-transform pairs defined in 
    PCA specs and uses the ThreeDPlotter class to generate plots for each brain region.

Run:
    python scripts/visualization/plot_pc_trajectories.py
"""

import pdb
import logging
from itertools import product
from collections import defaultdict

from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.psth_config import PSTHConfig
from socialgaze.config.interactivity_config import InteractivityConfig
from socialgaze.config.pca_config import PCAConfig
from socialgaze.config.plotting_config import PlottingConfig

from socialgaze.data.gaze_data import GazeData
from socialgaze.data.spike_data import SpikeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.features.psth_extractor import PSTHExtractor
from socialgaze.features.pc_projector import PCProjector
from socialgaze.visualization.pca_plotter import PCAPlotter

from socialgaze.specs.pca_specs import FIT_SPECS, TRANSFORM_SPECS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing config objects...")
    base_config = BaseConfig()
    fixation_config = FixationConfig()
    psth_config = PSTHConfig()
    interactivity_config = InteractivityConfig()
    pca_config = PCAConfig()
    plotting_config = PlottingConfig()

    logger.info("Initializing data managers...")
    gaze_data = GazeData(config=base_config)
    spike_data = SpikeData(config=base_config)

    logger.info("Initializing detectors...")
    fixation_detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    interactivity_detector = InteractivityDetector(config=interactivity_config)

    logger.info("Creating PSTH extractor...")
    psth_extractor = PSTHExtractor(
        config=psth_config,
        gaze_data=gaze_data,
        spike_data=spike_data,
        fixation_detector=fixation_detector,
        interactivity_detector=interactivity_detector,
    )

    logger.info("Creating PC projector and 3D plotter...")
    projector = PCProjector(config=pca_config, psth_extractor=psth_extractor)
    plotter = PCAPlotter(plotting_config=plotting_config, pca_config=pca_config)

    # available_keys = projector.get_available_fit_transform_region_keys()
    
    # for (fit_spec, transform_spec) in [(f, t) for f in FIT_SPECS for t in TRANSFORM_SPECS]:
    #     key = f"{fit_spec.name}__{transform_spec.name}"
    #     if key not in available_keys:
    #         continue
    #     for region in available_keys[key]:
    #         try:
    #             logger.info(f"Plotting: fit={fit_spec.name}, transform={transform_spec.name}, region={region}")                
    #             df, _ = projector.get_projection(fit_spec.name, transform_spec.name, region)
    #             save_path_static = plotter.plot_pc_trajectories_all_trials(df, fit_spec.name, transform_spec.name, region)
    #             logger.info(f"Saved static 3D plot to: {save_path_static}")

    #             save_path_rot = plotter.animate_pc_trajectories_3d(df, fit_spec.name, transform_spec.name, region)
    #             logger.info(f"Saved rotating 3D plot to: {save_path_rot}")

    #         except Exception as e:
    #             logger.warning(f"Plotting failed for {key} in {region}: {e}")

    # logger.info("PC projection plotting completed.")


    # === Comparison summary plots (only for selected fits + transforms) ===
    allowed_fits = {"fit_avg_face_obj", "fit_int_non_int_face_obj"}
    allowed_transforms = {"transform_avg_face_obj", "transform_int_non_int_face_obj"}

    for fit_spec in FIT_SPECS:
        if fit_spec.name not in allowed_fits:
            continue

        logger.info(f"\n[COMPARISON] Preparing plots for fit: {fit_spec.name}")
        comparison_dict = defaultdict(dict)
        region_set = set()

        for transform_spec in TRANSFORM_SPECS:
            if transform_spec.name not in allowed_transforms:
                continue

            key = f"{fit_spec.name}__{transform_spec.name}"
            available_regions = projector.get_available_fit_transform_region_keys().get(key, [])
            for region in available_regions:
                try:
                    df, _ = projector.get_projection(fit_spec.name, transform_spec.name, region)
                    results = projector.compare_category_trajectories(fit_spec.name, transform_spec.name, region)
                    _, meta = projector.get_projection(fit_spec.name, transform_spec.name, region)
                    comparison_dict[transform_spec.name][region] = {
                        "projection_df": df,
                        "comparison_metrics": results,
                        "projection_meta": meta
                    }
                    region_set.add(region)
                except Exception as e:
                    logger.warning(f"Comparison failed for {key} in {region}: {e}")

        if comparison_dict:
            plotter.plot_pc_trajectory_comparisons(comparison_dict, fit_spec.name, sorted(region_set))



if __name__ == "__main__":
    main()
