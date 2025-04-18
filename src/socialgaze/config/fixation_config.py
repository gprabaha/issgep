# src/socialgaze/config/fixation_config.py

from socialgaze.config.base_config import BaseConfig

class FixationConfig(BaseConfig):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)

        # Runtime behavior
        self.test_single_run: bool = False
        self.use_parallel: bool = False
        self.num_cpus: int = 8
        self.detect_fixations_again = False

        # HPC job parameters
        self.job_name: str = "fixation"
        self.partition: str = "day" if self.is_grace else "psych_day"
        self.env_name: str = "gaze_otnal" if self.is_grace else "gaze_processing"
        self.job_script_path: str = "scripts/analysis/01_fixation_detection.py"
        self.job_file_name: str = "fixation_job_array.txt"
        self.cpus_per_task: int = 1
        self.mem_per_cpu: int = 8000  # in MB
        self.time_limit: str = "00:15:00"
