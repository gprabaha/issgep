# src/socialgaze/features/crosscorr_calculator.py

from socialgaze.config.crosscorr_config import CrossCorrConfig
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector

class CrossCorrCalculator:
    def __init__(self,config: CrossCorrConfig, fixation_detector: FixationDetector, interactivity_detector: InteractivityDetector = None):
        self.config = config
        self.fixation_detector = fixation_detector
        self.interactivity_detector = interactivity_detector

        self.m1_face_m2_face_crosscorr_df: Optional[pd.DataFrame] = None
        self.m1_face_m2_face_shuffled_crosscorr_df: Optional[pd.DataFrame] = None
