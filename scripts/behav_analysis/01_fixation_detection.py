# scripts/behav_analysis/01_fixation_detection.py

"""
Script to detect fixations and saccades in eye-tracking data.

This script supports two modes:
1. Single-run mode (via --session, --run, and --agent) for detecting fixations locally.
2. Full job array submission for detecting fixations in parallel across sessions.

It also handles:
- Updating fixation and saccade ROI labels
- Reconciling label mismatches between fixations and saccades
- Saving and loading fixation/saccade dataframes
"""

import logging
import argparse

from socialgaze.config.fixation_config import FixationConfig
from socialgaze.data.gaze_data import GazeData
from socialgaze.features.fixation_detector import FixationDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Entry point for fixation detection.

    Command-line arguments:
        --session: Session name (e.g., '20240101')
        --run: Run number (e.g., '1')
        --agent: Agent name (e.g., 'm1' or 'm2')
    """
    parser = argparse.ArgumentParser(description="Run fixation detection.")
    parser.add_argument("--session", type=str, help="Session name (e.g., '20240101')")
    parser.add_argument("--run", type=str, help="Run number (e.g., '1')")
    parser.add_argument("--agent", type=str, help="Agent (e.g., 'm1' or 'm2')")
    args = parser.parse_args()

    # Load configuration and data
    fixation_config = FixationConfig()
    logger.info("Loaded FixationConfig.")
    gaze_data = GazeData(config=fixation_config)
    logger.info("Initialized GazeData.")
    detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    logger.info("Initialized FixationDetector.")

    # Mode 1: Single-run local detection
    if args.session and args.run and args.agent:
        logger.info("Running single-run fixation detection for session=%s, run=%s, agent=%s...",
                    args.session, args.run, args.agent)
        detector.detect_fixations_and_saccades_in_single_run(
            session_name=args.session,
            run_number=args.run,
            agent=args.agent
        )
        detector.save_dataframes()
        logger.info("Saved results for single-run.")
        return

    # Mode 2: Full dataset detection or label update
    if fixation_config.detect_fixations_again:
        logger.info("Submitting full fixation detection jobs to HPC...")
        detector.detect_fixations_through_hpc_jobs()
        detector.save_dataframes()
        logger.info("Saved fixation/saccade dataframes after HPC job.")
    else:
        logger.info("Loading previously saved fixation and saccade dataframes...")
        detector.load_dataframes()

    if fixation_config.update_labes_in_dfs:
        # Label and reconcile
        logger.info("Updating fixation ROI labels...")
        detector.update_fixation_locations()
        logger.info("Updating saccade ROI labels...")
        detector.update_saccade_from_to()
        logger.info("Reconciling mismatched fixation/saccade labels...")
        detector.reconcile_fixation_saccade_label_mismatches()
        # Save final results
        detector.save_dataframes()
        logger.info("Final dataframes saved successfully.")
    else:
        logger.info("Fixations and saccades already detected and labelled...")
    logger.info("Fixations df head:")
    logger.info(detector.fixations.head())
    logger.info("Saccades df head:")
    logger.info(detector.saccades.head())
    logger.info("Done.")


if __name__ == "__main__":
    main()
