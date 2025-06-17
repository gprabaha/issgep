# src/socialgaze/config/crosscorr_config.py

from socialgaze.config.fixation_config import FixationConfig
from socialgaze.utils.path_utils import get_crosscorr_job_file_path
from pathlib import Path


class CrossCorrConfig(FixationConfig):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)

        # Core parameters
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
        self.max_lag: int = 30000  # 30s at 1 kHz
        self.normalize: bool = True
        self.use_energy_norm: bool = True
        self.make_shuffle_stringent: bool = True
        self.num_shuffles: int = 500

        # HPC paths
        self.job_file_path: Path = get_crosscorr_job_file_path(self)
        self.worker_python_script_path: Path = (
            self.project_root / "scripts" / "behav_analysis" / "04_inter_agent_crosscorr.py"
        )
        self.env_name: str = "gaze_processing" if self.is_grace else "socialgaze"
