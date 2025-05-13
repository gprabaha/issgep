# scripts/visualization/plot_pc_trajectories.py

"""
Script: plot_pc_trajectories.py

Description:
    This script loads PC projections from disk and visualizes the top 3 PC dimensions 
    over time using 3D trajectories. It loops over all fit-transform pairs defined in 
    PCA specs and uses the ThreeDPlotter class to generate plots for each brain region.

Run:
    python scripts/visualization/pc_projection_plot.py
"""

import logging

from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.psth_config import PSTHConfig
from socialgaze.config.interactivity_config import InteractivityConfig
from socialgaze.config.plotting_config import PlottingConfig

from socialgaze.data.gaze_data import GazeData
from socialgaze.data.spike_data import SpikeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.features.psth_extractor import PSTHExtractor
from socialgaze.features.pc_projector import PCProjector
from socialgaze.visualization.3d_plotter import ThreeDPlotter

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
    projector = PCProjector(config=plotting_config, psth_extractor=psth_extractor)
    plotter = ThreeDPlotter(config=plotting_config)

    for fit_spec in FIT_SPECS:
        for transform_spec in TRANSFORM_SPECS:
            key = f"{fit_spec.name}__{transform_spec.name}"
            for region in projector.pc_projection_dfs.get(key, {}).keys():
                try:
                    logger.info(f"Plotting: fit={fit_spec.name}, transform={transform_spec.name}, region={region}")
                    df, _ = projector.get_projection(fit_spec.name, transform_spec.name, region)
                    plotter.plot_pc_trajectories_all_trials(df, fit_spec.name, transform_spec.name, region)
                except Exception as e:
                    logger.warning(f"Plotting failed for {key} in {region}: {e}")

    logger.info("PC projection plotting completed.")


if __name__ == "__main__":
    main()
