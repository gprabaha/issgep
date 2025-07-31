# scripts/behav_analysis/01_fixation_detection.py

"""
Script to detect fixations and saccades in eye-tracking data.

Supports:
1. Single-run mode (via --session, --run, and --agent) for local processing
2. Full job array submission for HPC-based detection

Also:
- Applies ROI labels
- Reconciles fixation/saccade label mismatches
- Generates and saves binary vector DataFrames (on disk, not in memory)
"""

import pdb
import logging
import argparse
import random

from socialgaze.config.fixation_config import FixationConfig
from socialgaze.data.gaze_data import GazeData
from socialgaze.features.fixation_detector import FixationDetector

from socialgaze.utils.loading_utils import load_df_from_pkl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run fixation detection.")
    parser.add_argument("--session", type=str, help="Session name (e.g., '20240101')")
    parser.add_argument("--run", type=str, help="Run number (e.g., '1')")
    parser.add_argument("--agent", type=str, help="Agent (e.g., 'm1' or 'm2')")
    args = parser.parse_args()

    # Load config and data
    fixation_config = FixationConfig()
    logger.info("Loaded FixationConfig.")
    gaze_data = GazeData(config=fixation_config)
    logger.info("Initialized GazeData.")
    detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    logger.info("Initialized FixationDetector.")

    # Single-run detection
    if args.session and args.run and args.agent:
        logger.info("Running single-run fixation detection for session=%s, run=%s, agent=%s...",
                    args.session, args.run, args.agent)
        detector.detect_fixations_and_saccades_in_single_run(
            session_name=args.session,
            run_number=args.run,
            agent=args.agent
        )
        logger.info("Saved results for single-run.")
        return

    # Full-job detection or label-only update
    if fixation_config.detect_fixations_again:
        logger.info("Submitting fixation detection jobs to HPC...")
        detector.detect_fixations_through_hpc_jobs()
        detector.save_dataframes()
        logger.info("Saved fixation/saccade dataframes after HPC job.")
    else:
        logger.info("Loading existing fixation and saccade dataframes...")
        detector.load_dataframes()

    if fixation_config.update_labes_in_dfs:
        logger.info("Updating fixation ROI labels...")
        detector.update_fixation_locations()
        logger.info("Updating saccade ROI labels...")
        detector.update_saccade_from_to()
        logger.info("Reconciling label mismatches...")
        detector.reconcile_fixation_saccade_label_mismatches()
    else:
        logger.info("Skipping ROI label updates...")

    # Add categories and save
    logger.info("Adding fixation category column...")
    detector.add_fixation_category_column()
    logger.info("Adding saccade category column...")
    detector.add_saccade_category_columns()
    logger.info("Get mutual out_of_roi fixations...")
    df = detector.get_mutual_out_of_roi_fixations()
    pdb.set_trace()

    detector.save_dataframes()
    logger.info("Saved final fixation and saccade dataframes.")

    # Generate binary vectors (on-disk only)
    logger.info("Generating and saving all binary vector dataframes...")
    detector.generate_and_save_binary_vectors()

    # Safely log summary info
    logger.info("Fixations df head:")
    if detector.fixations is not None and not detector.fixations.empty:
        logger.info(detector.fixations.head())
    else:
        logger.warning("Fixations dataframe is None or empty.")

    logger.info("Saccades df head:")
    if detector.saccades is not None and not detector.saccades.empty:
        logger.info(detector.saccades.head())
    else:
        logger.warning("Saccades dataframe is None or empty.")

    logger.info("Binary vector paths:")
    if hasattr(detector, "binary_vector_paths") and detector.binary_vector_paths:
        for btype, path in detector.binary_vector_paths.items():
            logger.info(f"{btype}: {path}")

        # Display head of one randomly chosen vector
        sampled_btype = random.choice(list(detector.binary_vector_paths.keys()))
        sampled_path = detector.binary_vector_paths[sampled_btype]
        try:
            df = load_df_from_pkl(sampled_path)
            logger.info(f"\nHead of binary vector DataFrame for {sampled_btype}:")
            logger.info(df.head())
        except Exception as e:
            logger.warning(f"Failed to load binary vector DataFrame for {sampled_btype}: {e}")

    else:
        logger.warning("No binary vectors generated or registered.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
