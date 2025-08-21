#!/usr/bin/env python3
# scripts/visualization/make_figure_4.py

"""
Script: make_figure_4.py

Description:
    Produces Figure 4-style visualizations using the PCProjectorPlotter:
      1) Interactive 3D view picking (press 'S' to cache view angles)
      2) One-row PC trajectory panels (uses cached views if available)
      3) Distance/Angle violin plots with pairwise stats across regions

Assumptions:
    - PCProjectorPlotter is defined at the bottom of
      socialgaze.features.pc_projector (same module as PCProjector)
    - PCAConfig provides: n_components, mean_center_for_angle,
      trajectories_dir, evr_bars_dir (and optionally view_cache_path)

Run examples:
    # Pick views then export both figure types for both comparisons:
    python scripts/visualization/make_figure_4.py --pick-views --export-row --export-violins

    # Only interactive/non-interactive face:
    python scripts/visualization/make_figure_4.py --comparison int_nonint_face --export-row

    # Face vs Object only, debug logs:
    python scripts/visualization/make_figure_4.py --comparison face_obj --pick-views --loglevel DEBUG
"""

import argparse
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

# IMPORTANT: this assumes you've added the subclass to the same module!
from socialgaze.features.pc_projector import PCProjectorPlotter


logger = logging.getLogger(__name__)


def _build_pipeline():
    """Initialize configs, data managers, detectors, and PSTH extractor."""
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

    return psth_extractor, pca_config


def run_for(plotter: PCProjectorPlotter, comparison: str, pick_views: bool, export_row: bool, export_violins: bool):
    """Run requested plotting actions for a single comparison."""
    if pick_views:
        logger.info(f"[{comparison}] Picking 3D views (press 'S' in each window to save)...")
        plotter.pick_3d_views_for(comparison)

    if export_row:
        logger.info(f"[{comparison}] Exporting single-row PC trajectory panel...")
        plotter.plot_pc_timeseries_row(comparison=comparison, export_pdf=True)

    if export_violins:
        logger.info(f"[{comparison}] Exporting distance/angle violin plots...")
        plotter.plot_violin_distance_angle(comparison=comparison, export_pdf=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make Figure 4: 3D PC trajectories with cached views and violin plots."
    )
    parser.add_argument(
        "--comparison",
        choices=["face_obj", "int_non_int_face", "both"],
        default="both",
        help="Which comparison(s) to process."
    )
    parser.add_argument(
        "--pick-views",
        action="store_true",
        help="Interactively pick and cache 3D views (press 'S' to save)."
    )
    parser.add_argument(
        "--export-row",
        action="store_true",
        help="Export single-row PC trajectory panel (uses cached views if present)."
    )
    parser.add_argument(
        "--export-violins",
        action="store_true",
        help="Export distance/angle violin plots with pairwise stats."
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))
    logger.info("== Building plotting pipeline ==")

    psth_extractor, pca_config = _build_pipeline()

    # The PCProjectorPlotter uses pca_config for directories & PCA params
    plotter = PCProjectorPlotter(
        psth_extractor=psth_extractor,
        pca_config=pca_config,
    )

    if args.comparison in ("face_obj", "int_non_int_face"):
        run_for(plotter, args.comparison, args.pick_views, args.export_row, args.export_violins)
    else:
        # both
        run_for(plotter, "face_obj", args.pick_views, args.export_row, args.export_violins)
        run_for(plotter, "int_non_int_face", args.pick_views, args.export_row, args.export_violins)

    logger.info("== Done. ==")


if __name__ == "__main__":
    main()
