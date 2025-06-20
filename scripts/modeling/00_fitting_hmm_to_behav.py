# scripts/modeling/00_fitting_hmm__behav.py

import pdb
import logging

from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.interactivity_config import InteractivityConfig
from socialgaze.config.crosscorr_config import CrossCorrConfig
from socialgaze.config.hmm_config import HMMConfig

from socialgaze.data.gaze_data import GazeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.features.crosscorr_calculator import CrossCorrCalculator
from socialgaze.models.hmm_fitter import HMMFitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load configs
    fixation_config = FixationConfig()
    interactivity_config = InteractivityConfig()
    crosscorr_config = CrossCorrConfig()
    model_config = HMMConfig()

    # Initialize data + detectors
    fixation_detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    interactivity_detector = InteractivityDetector(config=interactivity_config)
    crosscorr_calculator = CrossCorrCalculator(
        config=crosscorr_config,
        fixation_detector=fixation_detector,
        interactivity_detector=interactivity_detector,
    )

    modeler = HMMFitter(model_config, fixation_detector, crosscorr_calculator, interactivity_detector)