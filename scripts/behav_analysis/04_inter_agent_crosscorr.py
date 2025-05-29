# scripts/behav_analysis/04_inter_agent_crosscorr.py

from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.interactivity_config import InteractivityConfig
from socialgaze.config.crosscorr_config import CrossCorrConfig


from socialgaze.data.gaze_data import GazeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.features.crosscorr_calculator import CrossCorrCalculator


def main():
    # Load all configs
    base_config = BaseConfig()
    fixation_config = FixationConfig()
    fix_prob_config = FixProbConfig()
    interactivity_config = InteractivityConfig()

    # Initialize core data and detectors
    gaze_data = GazeData(config=base_config)
    fixation_detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    interactivity_detector = InteractivityDetector(config=interactivity_config)

    # Initialize fixation probability detector
    crosscorr_calculator = CrossCorrCalculator(
        config=crosscorr_config,
        fixation_detector=fixation_detector,
        interactivity_detector=interactivity_detector
    )


if __name__ == "__main__":
    main()