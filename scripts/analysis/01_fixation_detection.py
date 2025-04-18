# scripts/analysis/01_fixation_detection.py

import logging
import argparse

from socialgaze.config.fixation_config import FixationConfig
from socialgaze.data.gaze_data import GazeData
from socialgaze.features.fixation_detector import FixationDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run fixation detection.")

    parser.add_argument("--session", type=str, help="Session name (e.g., '20240101')")
    parser.add_argument("--run", type=str, help="Run number (e.g., '1')")
    parser.add_argument("--agent", type=str, help="Agent (e.g., 'm1' or 'm2')")

    args = parser.parse_args()

    fixation_config = FixationConfig()
    gaze_data = GazeData(config=fixation_config)
    detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)

    if args.session and args.run and args.agent:
        logger.info("Running single-run fixation detection...")
        detector.detect_fixations_and_saccades_in_single_run(
            session_name=args.session,
            run_number=args.run,
            agent=args.agent,
            config=fixation_config
        )
    else:
        if fixation_config.detect_fixations_again:
            logger.info("Running full fixation detection job submission...")
            detector.detect_fixations_through_hpc_jobs(fixation_config)
            # detector.update_fixation_locations()
            # detector.update_saccade_from_to()
            detector.save_dataframes()
        else:
            detector.load_dataframes()
            detector.update_fixation_locations()
            detector.update_saccade_from_to()
            detector.save_dataframes()

if __name__ == "__main__":
    main()
