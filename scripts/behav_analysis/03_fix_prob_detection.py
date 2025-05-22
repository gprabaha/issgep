from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.fix_prob_config import FixProbConfig

from socialgaze.data.gaze_data import GazeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.fix_prob_detector import FixProbDetector

def main():
    # Initialize base and fixation configs
    base_config = BaseConfig()
    fixation_config = FixationConfig()
    fix_prob_config = FixProbConfig()  # Inherits from FixationConfig

    # Initialize gaze data and fixation detector
    gaze_data = GazeData(config=base_config)
    detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)

    # Compute fixation probabilities
    fix_prob_detector = FixProbDetector(fixation_detector=detector, config=fix_prob_config)
    fix_prob_df = fix_prob_detector.compute_and_save()

    # Optional: if you want to load later instead
    # fix_prob_df = fix_prob_detector.get_data()

    # Preview the output
    print(fix_prob_df.head(20))


if __name__ == "__main__":
    main()
