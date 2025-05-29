# src/socialgaze/config/crosscorr_config.py

from socialgaze.config.fixation_config import FixationConfig

class CrossCorrConfig(FixationConfig):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)