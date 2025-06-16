# src/socialgaze/config/crosscorr_config.py

from socialgaze.config.fixation_config import FixationConfig

class CrossCorrConfig(FixationConfig):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)
        self.use_parallel: bool = True
        self.show_inner_tqdm: bool = True
        self.binary_vector_types_to_use: list = (
            "face_fixation",
            "saccade_to_face",
            "saccade_from_face"
        )
        self.crosscorr_agent_behavior_pairs = [
            ("m1", "face_fixation", "m2", "face_fixation"),
            ("m1", "face_fixation", "m2", "saccade_to_face"),
            ("m1", "face_fixation", "m2", "saccade_from_face"),
            ("m2", "face_fixation", "m1", "saccade_to_face"),
            ("m2", "face_fixation", "m1", "saccade_from_face"),
        ]
        self.max_lag: int = 30000  # lag in frames: 30 seconds since data is at 1khz
        self.normalize: bool = True
        self.use_energy_norm: bool = True
