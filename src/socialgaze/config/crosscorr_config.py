# src/socialgaze/config/crosscorr_config.py

from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.path_utils import CrossCorrPaths, get_job_out_dir, get_log_dir, get_sbatch_script_path


class CrossCorrConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # === General execution ===
        self.use_parallel: bool = True
        self.show_inner_tqdm: bool = True
        self.run_single_test_case: bool = False

        # === Binary vector + agent/behavior pair config ===
        self.binary_vector_types_to_use: list = (
            "face_fixation",
            "saccade_from_face",
        )
        self.crosscorr_agent_behavior_pairs = [
            ("m1", "face_fixation", "m2", "face_fixation"),
            ("m1", "face_fixation", "m2", "saccade_to_face"),
            ("m1", "face_fixation", "m2", "saccade_from_face"),
            ("m2", "face_fixation", "m1", "saccade_to_face"),
            ("m2", "face_fixation", "m1", "saccade_from_face"),
        ]

        # === Cross-correlation parameters ===
        self.max_lag: int = 30000  # 30 sec at 1 kHz
        self.normalize: bool = True
        self.use_energy_norm: bool = False

        # === Shuffle + smoothing ===
        self.num_shuffles: int = 1000
        self.make_shuffle_stringent: bool = True
        self.do_smoothing: bool = False
        self.smoothing_sigma_n_bins: int = 2

        # === HPC job configuration ===
        self.job_name = "crosscorr_shuffled"
        self.job_file_name = "crosscorr_shuffled_job_array.txt"
        self.python_script_relative = "scripts/behav_analysis/04_inter_agent_crosscorr.py"
        self.env_name = "socialgaze" if self.is_grace else "gaze_processing"
        self.partition = "day" if self.is_grace else "psych_day"
        self.cpus_per_task = 8
        self.mem_per_cpu = 4000
        self.time_limit = "00:05:00"

        # === Paths ===
        self.paths = CrossCorrPaths(self)
        self._assign_paths()

    def _assign_paths(self):
        self.job_file_path = self.paths.get_job_file_path(self.job_file_name)
        self.job_out_dir = get_job_out_dir(self.project_root)
        self.log_dir = get_log_dir(self.job_out_dir)
        self.worker_python_script_path = self.paths.get_worker_script_path()
        self.sbatch_script_path = get_sbatch_script_path(self.job_out_dir, self.job_name)
