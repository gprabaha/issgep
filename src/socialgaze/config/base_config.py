# src/socialgaze/config/base_config.py


from pathlib import Path
import logging
from typing import Dict, Any, Optional, List
import pandas as pd

from socialgaze.config.environment import detect_runtime_environment
from socialgaze.utils.loading_utils import load_df_from_pkl, load_config_from_json
from socialgaze.utils.path_utils import (
    determine_root_data_dir,
    get_pupil_file_path,
    get_roi_file_path,
    get_time_file_path
)
from socialgaze.utils.discovery_utils import find_valid_sessions, filter_sessions_with_ephys
from socialgaze.utils.saving_utils import save_config_to_json
from socialgaze.utils.conversion_utils import object_to_dict, assign_dict_attributes_to_object

import pdb

logger = logging.getLogger(__name__)


class BaseConfig:
    """
    A configuration manager for handling paths, environment flags, and available session/run combinations.
    It automatically detects runtime environments and verifies the existence of all necessary data files.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the BaseConfig object. If a saved config is provided, load it.
        Otherwise, auto-detect the environment, set paths, and discover valid sessions/runs.

        Args:
            config_path (Optional[str]): Path to a saved JSON config file.
        """

        # Detect runtime environment
        env = detect_runtime_environment()
        self.is_cluster = env["is_cluster"]
        self.is_grace = env["is_grace"]
        self.prabaha_local = env["prabaha_local"]
       

        # Set core project directories
        self.project_root = Path(__file__).resolve().parents[3]

        self.config_folder = self.project_root / "src/socialgaze/config/saved_config/"
        self.filename = self.get_config_filename()
        self.config_path = self.config_folder / self.filename

        self.processed_data_dir = self.project_root / "data/processed"
        self.output_dir = self.project_root / "outputs"
        self.plots_dir = self.output_dir / "plots"

        # Supported behavioral data types
        self.behav_data_types = ['positions', 'roi_vertices', 'pupil', 'neural_timeline']

        # Raw data directories
        self.data_dir = Path(determine_root_data_dir(
            is_cluster=self.is_cluster,
            is_grace=self.is_grace,
            prabaha_local=self.prabaha_local
        ))
        self.position_dir = self.data_dir / "eyetracking/aligned_raw_samples/position"
        self.time_dir = self.data_dir / "eyetracking/aligned_raw_samples/time"
        self.pupil_dir = self.data_dir / "eyetracking/aligned_raw_samples/pupil_size"
        self.roi_dir = self.data_dir / "eyetracking/roi_rects"

        # Pattern used for parsing filenames
        self.file_pattern = "{session_date}_position_{run_number}.mat"

        # Session and run metadata
        self.session_names: List[str] = []
        self.runs_by_session: Dict[str, List[str]] = {}

        if config_path:
            self.load_from_file(config_path)
        else:
            self.initialize_sessions_and_runs()
            ephys_days_and_monkeys_filepath = self.processed_data_dir / "ephys_days_and_monkeys.pkl"
            ephys_days_and_monkeys_df = load_df_from_pkl(ephys_days_and_monkeys_filepath)
            self.extract_sessions_with_ephys_data(ephys_days_and_monkeys_df)
            self.save_to_file(self.config_path)
            logger.info(f"base_config generated and saved to {self.config_path}")

    # -----------------------------
    # Session / run discovery
    # -----------------------------

    def get_config_filename(self) -> str:
        if self.is_cluster: 
            if self.is_grace:
                return "grace_config.json"
            else:
                return "milgram_config.json"
        else:
            return "local_config.json"


    def initialize_sessions_and_runs(self) -> None:
        """
        Detects valid session/run pairs by checking if all required files exist
        (time, pupil, ROI) alongside each position file.
        """
        self.session_names, self.runs_by_session = find_valid_sessions(self,
            path_fns={
                'time': get_time_file_path,
                'pupil': get_pupil_file_path,
                'roi': get_roi_file_path,
            }
        )

    def extract_sessions_with_ephys_data(self, ephys_days_and_monkeys_df: pd.DataFrame) -> None:
        """
        Filters out sessions that do not have corresponding electrophysiology data.

        Args:
            ephys_days_and_monkeys_df (pd.DataFrame): DataFrame with valid ephys session names.
        """
        self.session_names, self.runs_by_session = filter_sessions_with_ephys(
            self.session_names,
            self.runs_by_session,
            ephys_days_and_monkeys_df
        )

    # -----------------------------
    # Save / load
    # -----------------------------

    def save_to_file(self, config_path: str) -> None:
        """
        Saves the current configuration to a JSON file.

        Args:
            config_path (str): Destination path for the config JSON.
        """
        save_config_to_json(object_to_dict(self), config_path)

    def load_from_file(self, config_path: str) -> None:
        """
        Loads configuration attributes from a JSON file.

        Args:
            config_path (str): Path to the saved config JSON.
        """
        config_data = load_config_from_json(config_path)
        assign_dict_attributes_to_object(self, config_data)
