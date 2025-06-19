# src/socialgaze/config/crosscorr_config.py

from socialgaze.config.fixation_config import FixationConfig
from socialgaze.utils.path_utils import (
    get_crosscorr_job_file_path,
    get_crosscorr_shuffled_output_dir,
    get_crosscorr_worker_script_path,
    get_job_file_path,
    get_sbatch_script_path,
    get_job_out_dir,
    get_log_dir,
)
from pathlib import Path


class CrossCorrConfig(FixationConfig):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)

        # === Analysis parameters ===
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
        self.max_lag: int = 30000  # 30 seconds at 1 kHz
        self.normalize: bool = True
        self.use_energy_norm: bool = True
        self.make_shuffle_stringent: bool = True
        self.num_shuffles: int = 500
        self.run_single_test_case: bool = True

        # === Output paths ===
        self.crosscorr_shuffled_output_dir = get_crosscorr_shuffled_output_dir(self)
        self.crosscorr_shuffled_temp_dir = self.crosscorr_shuffled_output_dir / "temp"
        self.crosscorr_shuffled_temp_dir.mkdir(parents=True, exist_ok=True)

        # === HPC job config (overriding fixation config) ===
        self.job_name = "crosscorr_shuffled"
        self.job_file_name = "crosscorr_shuffled_job_array.txt"
        self.python_script_relative = "scripts/behav_analysis/04_inter_agent_crosscorr.py"
        self.env_name = "socialgaze" if self.is_grace else "gaze_processing"
        self.partition = "day" if self.is_grace else "psych_day"
        self.cpus_per_task = 8
        self.mem_per_cpu = 4000
        self.time_limit = "00:20:00"

        # Assign paths specific to this job
        self._assign_paths()

    def _assign_paths(self):
        self.job_file_path = get_job_file_path(self.project_root, self.job_file_name)
        self.job_out_dir = get_job_out_dir(self.project_root)
        self.log_dir = get_log_dir(self.job_out_dir)
        self.worker_python_script_path = get_crosscorr_worker_script_path(self.project_root)
        self.sbatch_script_path = get_sbatch_script_path(self.job_out_dir, self.job_name)
