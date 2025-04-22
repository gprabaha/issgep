# src/socialgaze/config/fixation_config.py

import logging
from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.path_utils import (
    get_fixation_config_json_path,
    get_fix_binary_vec_df_path,
    get_job_file_path,
    get_fixation_temp_dir,
    get_fixation_df_path,
    get_saccade_df_path,
    get_job_out_dir,
    get_log_dir,
    get_worker_python_script_path,
    get_sbatch_script_path
)
from socialgaze.utils.saving_utils import save_config_to_json
from socialgaze.utils.loading_utils import load_config_from_json
from socialgaze.utils.conversion_utils import object_to_dict, assign_dict_attributes_to_object
from socialgaze.utils.discovery_utils import get_num_available_cpus

logger = logging.getLogger(__name__)


class FixationConfig(BaseConfig):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)

        # === Behavior flags ===
        self.test_single_run = False
        self.detect_fixations_again = True
        self.update_labes_in_dfs = True
        self.fixation_labeling_log_interval = 100

        # === HPC job config ===
        self.job_name = "fixation"
        self.partition = "day" if self.is_grace else "psych_day"
        self.env_name = "gaze_otnal" if self.is_grace else "gaze_processing"
        self.python_script_relative = "scripts/behav_analysis/01_fixation_detection.py"
        self.job_file_name = "fixation_job_array.txt"
        self.cpus_per_task = 1
        self.mem_per_cpu = 8000
        self.time_limit = "00:15:00"

        self._assign_paths()
        self.save_to_json()

    def _assign_paths(self):
        self.fixation_config_path = get_fixation_config_json_path(self.config_folder)
        self.temp_dir = get_fixation_temp_dir(self.processed_data_dir)
        self.fixation_df_path = get_fixation_df_path(self.processed_data_dir)
        self.saccade_df_path = get_saccade_df_path(self.processed_data_dir)
        self.fix_binary_vec_df_path = get_fix_binary_vec_df_path(self)
        self.job_file_path = get_job_file_path(self.project_root, self.job_file_name)
        self.job_out_dir = get_job_out_dir(self.project_root)
        self.log_dir = get_log_dir(self.job_out_dir)
        self.worker_python_script_path = get_worker_python_script_path(self.project_root, self.python_script_relative)
        self.sbatch_script_path = get_sbatch_script_path(self.job_out_dir, self.job_name)

    def to_dict(self) -> dict:
        return {
            "test_single_run": self.test_single_run,
            "detect_fixations_again": self.detect_fixations_again,
            "update_labes_in_dfs": self.update_labes_in_dfs,
            "fixation_labeling_log_interval": self.fixation_labeling_log_interval,
            "job_name": self.job_name,
            "partition": self.partition,
            "env_name": self.env_name,
            "python_script_relative": self.python_script_relative,
            "job_file_name": self.job_file_name,
            "cpus_per_task": self.cpus_per_task,
            "mem_per_cpu": self.mem_per_cpu,
            "time_limit": self.time_limit,
            "fixation_type_to_process": self.fixation_type_to_process
        }

    def save_to_json(self):
        save_config_to_json(object_to_dict(self), self.fixation_config_path)
        logger.info("Saved fixation config to %s", self.fixation_config_path)

    @classmethod
    def load_from_json(cls, base_config: BaseConfig):
        load_path = base_config.fixation_config_path
        config_data = load_config_from_json(load_path)
        fixation_config = cls()
        fixation_config.__dict__.update(base_config.__dict__)
        fixation_config._update_from_dict(config_data)
        return fixation_config

    def _update_from_dict(self, cfg_dict: dict):
        assign_dict_attributes_to_object(self, cfg_dict)
        self._assign_paths()