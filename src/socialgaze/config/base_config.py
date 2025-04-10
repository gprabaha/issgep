# src/socialgaze/config/base_config.py

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from socialgaze.config.environment import detect_runtime_environment
from socialgaze.utils.discovery_utils import find_valid_sessions, filter_sessions_with_ephys
from socialgaze.utils.loading_utils import load_df_from_pkl
from socialgaze.utils.path_utils import (
    determine_root_data_dir,
    get_pupil_file_path,
    get_roi_file_path,
    get_time_file_path
)


import pdb


logger = logging.getLogger(__name__)


class BaseConfig:
    """
    A configuration manager for handling paths, environment flags, and available session/run combinations.
    It automatically detects runtime environments and verifies the existence of data files.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the BaseConfig object. Detects environment, sets up paths,
        and optionally loads configuration from a JSON file.

        Args:
            config_path (Optional[str]): Path to a saved JSON config file. If provided, config is loaded from this path.
        """
        
        env = detect_runtime_environment()
        self.is_cluster = env["is_cluster"]
        self.is_grace = env["is_grace"]
        self.prabaha_local = env["prabaha_local"]

        self.project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
        self.processed_data_dir = self.project_root / "data/processed"
        self.output_dir = self.project_root / "outputs"
        self.plots_dir = self.output_dir / "plots"

        self.behav_data_types = ['positions', 'roi_vertices', 'pupil', 'neural_timeline']

        self.data_dir = Path(determine_root_data_dir(
            is_cluster=self.is_cluster,
            is_grace=self.is_grace,
            prabaha_local=self.prabaha_local
        ))
        self.position_dir = self.data_dir / "eyetracking/aligned_raw_samples/position"
        self.time_dir = self.data_dir / "eyetracking/aligned_raw_samples/time"
        self.pupil_dir = self.data_dir / "eyetracking/aligned_raw_samples/pupil_size"
        self.roi_dir = self.data_dir / "eyetracking/roi_rects"
        self.file_pattern = "{session_date}_position_{run_number}.mat"

        self.session_names: List[str] = []
        self.runs_by_session: Dict[str, List[str]] = {}

        if config_path:
            self.load_from_file(config_path)
        else:
            self.initialize_sessions_and_runs()
            ephys_days_and_monkeys_filepath = self.processed_data_dir / "ephys_days_and_monkeys.pkl"
            ephys_days_and_monkeys_df = load_df_from_pkl(ephys_days_and_monkeys_filepath)
            self.extract_sessions_with_ephys_data(ephys_days_and_monkeys_df)


    # -----------------------------
    # Path logic
    # -----------------------------

    def determine_root_data_dir(self) -> str:
        """
        Determines the root directory of raw data based on the current runtime environment.

        Returns:
            str: Path to the root data directory.
        """
        if self.is_cluster:
            return "/gpfs/gibbs/project/chang/pg496/data_dir/social_gaze/" if self.is_grace \
                else "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/"
        if self.prabaha_local:
            return "/Users/prabaha/data_dir/social_gaze"
        return "../data/raw"


    # -----------------------------
    # Session / run discovery
    # -----------------------------

    def initialize_sessions_and_runs(self) -> None:
        self.session_names, self.runs_by_session = find_valid_sessions(
            self.position_dir, path_fns={
                'time': get_time_file_path,
                'pupil': get_pupil_file_path,
                'roi': get_roi_file_path,
            }
        )

    def extract_sessions_with_ephys_data(self, ephys_days_and_monkeys_df: pd.DataFrame) -> None:
        self.session_names, self.runs_by_session = filter_sessions_with_ephys(
            self.session_names, self.runs_by_session, ephys_days_and_monkeys_df
        )
    

    # -----------------------------
    # Save / load
    # -----------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the config object for serialization.

        Returns:
            Dict[str, Any]: Dictionary with config values.
        """
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
            if not key.startswith('_')
        }


    def save_to_file(self, config_path: str) -> None:
        """
        Saves the current configuration to a JSON file.

        Args:
            config_path (str): Path where the config file should be saved.
        """
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


    def load_from_file(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        for key, value in config_data.items():
            if isinstance(value, str) and ("dir" in key or "path" in key):
                setattr(self, key, Path(value))
            else:
                setattr(self, key, value)

