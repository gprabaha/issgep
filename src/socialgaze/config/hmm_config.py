# src/socialgaze/config/hmm_config.py

from socialgaze.config.base_config import BaseConfig

class HMMConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # HMM-specific options
        self.binary_vector_types_to_use: list = [
            "face_fixation",
            "saccade_to_face",
            "saccade_from_face"
        ]

        self.num_states: int = 5
        self.model_type: str = "categorical"  # or "bernoulli", "gaussian", etc. if needed in future
        self.remake_hmm_fits: bool = True

        # Paths for saving/loading
        self.hmm_model_output_path = self.root_output_dir / "models" / "hmm"
        self.hmm_model_output_path.mkdir(parents=True, exist_ok=True)
