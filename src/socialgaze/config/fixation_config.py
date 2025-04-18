# src/socialgaze/config/fixation_config.py

import logging
import json
from pathlib import Path
from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.path_utils import (
    get_fixation_config_json_path,
    get_job_file_path,
    get_fixation_temp_dir,
    get_fixation_df_path,
    get_saccade_df_path,
    get_job_out_dir,
    get_log_dir,
    get_worker_python_script_path,
    get_sbatch_script_path
)

logger = logging.getLogger(__name__)


class FixationConfig(BaseConfig):
    """
    A configuration class that extends BaseConfig with fixation-specific options and HPC job parameters.
    Includes path resolution using path_utils and support for JSON save/load.
    """

    def __init__(self, config_path: str = None):
        super().__init__(config_path)

        # === Behavior flags ===
        self.test_single_run: bool = False
        self.detect_fixations_again: bool = True
        self.update_labes_in_dfs: bool = True
        self.fixation_labeling_log_interval: int = 100

        # === HPC job config ===
        self.job_name: str = "fixation"
        self.partition: str = "day" if self.is_grace else "psych_day"
        self.env_name: str = "gaze_otnal" if self.is_grace else "gaze_processing"
        self.python_script_relative: str = "scripts/analysis/01_fixation_detection.py"
        self.job_file_name: str = "fixation_job_array.txt"
        self.cpus_per_task: int = 1
        self.mem_per_cpu: int = 8000  # in MB
        self.time_limit: str = "00:15:00"

        # === Computed paths ===
        self._assign_paths()
        self.save_to_json()


    def _assign_paths(self):
        """Assigns all path attributes based on current config values."""
        self.fixation_config_path = get_fixation_config_json_path(self.project_root)
        self.temp_dir = get_fixation_temp_dir(self.processed_data_dir)
        self.fixation_df_path = get_fixation_df_path(self.processed_data_dir)
        self.saccade_df_path = get_saccade_df_path(self.processed_data_dir)
        self.job_file_path = get_job_file_path(self.project_root, self.job_file_name)
        self.job_out_dir = get_job_out_dir(self.project_root)
        self.log_dir = get_log_dir(self.job_out_dir)
        self.worker_python_script_path = get_worker_python_script_path(self.project_root, self.python_script_relative)
        self.sbatch_script_path = get_sbatch_script_path(self.job_out_dir, self.job_name)

    def to_dict(self) -> dict:
        """
        Returns a dictionary containing only the save-worthy parameters (not computed paths).

        Returns:
            dict: Dictionary of config attributes.
        """
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
            "time_limit": self.time_limit
        }

    def update_from_dict(self, cfg_dict: dict):
        """
        Updates this config object from a dictionary of values.

        Args:
            cfg_dict (dict): Dictionary of values to update from.
        """
        for k, v in cfg_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._assign_paths()

    def save_to_json(self):
        """
        Saves the fixation config to a default JSON path based on project_root.
        """
        save_path = self.fixation_config_path
        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        logger.info("Saved fixation config to %s", save_path)

    @classmethod
    def load_from_json(cls, base_config: BaseConfig):
        """
        Loads fixation config from the default JSON path and merges it with base_config.

        Args:
            base_config (BaseConfig): The base config object (already populated).

        Returns:
            FixationConfig: Fully populated FixationConfig instance.
        """
        load_path = self.fixation_config_path
        with open(load_path, "r") as f:
            data = json.load(f)
        fix_cfg = cls()
        fix_cfg.__dict__.update(base_config.__dict__)  # Inherit base config
        fix_cfg.update_from_dict(data)
        return fix_cfg

