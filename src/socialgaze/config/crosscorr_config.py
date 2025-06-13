# src/socialgaze/config/crosscorr_config.py

from socialgaze.config.fixation_config import FixationConfig

class CrossCorrConfig(FixationConfig):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)
        self.binary_vector_types_to_use: list = (
            "face_fixation",
            "saccade_to_face",
            "saccade_from_face"
        )
        self.max_lag: int = 30000  # lag in frames: 30 seconds since data is at 1khz
        self.normalize: bool = True
        self.use_energy_norm: bool = True
