# src/socialgaze/config/fixation_config.py


from pathlib import Path
from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.path_utils import (
    get_fixation_temp_dir,
    get_job_file_path,
    get_job_script_path,
    get_fixation_df_path,
    get_saccade_df_path
)


class FixationConfig(BaseConfig):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)

        # Runtime behavior
        self.test_single_run: bool = False
        self.use_parallel: bool = False
        self.num_cpus: int = 8
        self.detect_fixations_again = False
        self.update_labes_in_dfs = False
        self.fixation_labeling_log_interval = 100

        # HPC job parameters
        self.job_name: str = "fixation"
        self.partition: str = "day" if self.is_grace else "psych_day"
        self.env_name: str = "gaze_otnal" if self.is_grace else "gaze_processing"
        self.job_script_relative: str = "scripts/analysis/01_fixation_detection.py"
        self.job_file_name: str = "fixation_job_array.txt"
        self.cpus_per_task: int = 1
        self.mem_per_cpu: int = 8000
        self.time_limit: str = "00:15:00"

        # Assigned paths (computed using utils)
        self.temp_dir: Path = get_fixation_temp_dir(self.processed_data_dir)
        self.job_file_path: Path = get_job_file_path(self.project_root, self.job_file_name)
        self.job_script_path: Path = get_job_script_path(self.project_root, self.job_script_relative)
        self.fixation_df_path: Path = get_fixation_df_path(self.processed_data_dir)
        self.saccade_df_path: Path = get_saccade_df_path(self.processed_data_dir)

