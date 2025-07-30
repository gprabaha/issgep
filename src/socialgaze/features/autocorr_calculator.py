

from socialgaze.config.crosscorr_config import CrossCorrConfig
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.utils.hpc_utils import (
    generate_crosscorr_job_file,
    submit_dsq_array_job,
    track_job_completion
)

from socialgaze.utils.path_utils import AutoCorrPaths
from socialgaze.utils.saving_utils import save_df_to_pkl



class AutoCorrCalculator:
    
    def __init__(self, config: CrossCorrConfig, fixation_detector: FixationDetector, interactivity_detector: InteractivityDetector):
        self.config = config
        self.fixation_detector = fixation_detector
        self.interactivity_detector = interactivity_detector
        self.paths = AutoCorrPaths(self.config)
        
    
