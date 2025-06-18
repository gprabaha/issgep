# scripts/behav_analysis/04_inter_agent_crosscorr.py

import logging
import argparse

from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.interactivity_config import InteractivityConfig
from socialgaze.config.crosscorr_config import CrossCorrConfig

from socialgaze.data.gaze_data import GazeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.features.crosscorr_calculator import CrossCorrCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str)
    parser.add_argument("--run", type=str)
    parser.add_argument("--a1", type=str)
    parser.add_argument("--b1", type=str)
    parser.add_argument("--a2", type=str)
    parser.add_argument("--b2", type=str)
    parser.add_argument("--mode", choices=["standard", "shuffled"], default="standard")
    parser.add_argument("--period_type", type=str, default="full")

    args = parser.parse_args()

    # Load configs
    base_config = BaseConfig()
    fixation_config = FixationConfig()
    interactivity_config = InteractivityConfig()
    crosscorr_config = CrossCorrConfig()

    # Initialize data + detectors
    gaze_data = GazeData(config=base_config)
    fixation_detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    interactivity_detector = InteractivityDetector(config=interactivity_config)

    calculator = CrossCorrCalculator(
        config=crosscorr_config,
        fixation_detector=fixation_detector,
        interactivity_detector=interactivity_detector,
    )

    if args.session and args.run and args.a1 and args.b1 and args.a2 and args.b2:
        if args.mode == "shuffled":
            args.run = int(args.run)
            calculator.compute_shuffled_crosscorrelations_for_single_run(
                session=args.session,
                run=args.run,
                a1=args.a1, b1=args.b1,
                a2=args.a2, b2=args.b2,
                period_type=args.period_type,
            )
        else:
            raise NotImplementedError("Only shuffled mode is supported for single-run jobs.")
    else:
        # calculator.compute_crosscorrelations(by_interactivity_period=False)
        # calculator.compute_crosscorrelations(by_interactivity_period=True)
        calculator.compute_shuffled_crosscorrelations(by_interactivity_period=False)
        calculator.compute_shuffled_crosscorrelations(by_interactivity_period=True)


if __name__ == "__main__":
    main()
