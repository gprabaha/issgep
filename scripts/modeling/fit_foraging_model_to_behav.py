# scripts/modeling/fit_foraging_model_to_behav.py

import pdb
import logging

from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.interactivity_config import InteractivityConfig
from socialgaze.config.foraging_config import ForagingConfig

from socialgaze.data.gaze_data import GazeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector

from socialgaze.models.foraging_modeler import ForagingModeler

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting foraging model pipeline...")

    # === Load configs ===
    base_config = BaseConfig()
    fixation_config = FixationConfig()
    interactivity_config = InteractivityConfig()
    foraging_config = ForagingConfig()

    # === Initialize data and detectors ===
    gaze_data = GazeData(config=base_config)
    fixation_detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    interactivity_detector = InteractivityDetector(config=interactivity_config)
    
    fixation_df = fixation_detector.get_behavior_data("fixations")
    saccade_df = fixation_detector.get_behavior_data("saccades")
    inter_df = interactivity_detector.load_interactivity_periods()
    pdb.set_trace()

    # === Initialize and run modeler ===
    modeler = ForagingModeler(
        config=foraging_config,
        fixation_detector=fixation_detector,
        interactivity_detector=interactivity_detector
    )

    modeler.compute_fixation_bout_metrics(by_interactivity_period=foraging_config.include_interactivity)

    logger.info("Foraging modeling complete.")


'''

# Load your data
# fixation_df should already be loaded as shown in your example

# Basic single model approach
from semi_markov_model import SemiMarkovModel

model = SemiMarkovModel()
model.fit(fixation_df, distribution_type='lognorm')
model.get_model_summary()
model.plot_transition_matrix()
model.plot_dwell_distributions()

'''

'''
# For comparing M1 and M2
from multi_agent_analysis import MultiAgentSemiMarkovAnalysis

analyzer = MultiAgentSemiMarkovAnalysis()
analyzer.fit_agent_models(fixation_df, agents=['m1', 'm2'])

# Compare the agents
analyzer.compare_transition_matrices()
dwell_stats = analyzer.compare_dwell_times()
test_results = analyzer.statistical_tests()  # Statistical tests between m1 and m2
sequence_metrics = analyzer.plot_sequence_metrics(fixation_df)

'''

if __name__ == "__main__":
    main()
