# src/socialgaze/config/base_config.py

import os
from pathlib import Path
import logging
import socket
import getpass
from typing import Dict, Optional, List
import pandas as pd

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
        Initialize the BaseConfig object. If a saved config is provided and exists, load it.
        Otherwise, auto-detect the environment, set paths, and discover valid sessions/runs.

        Args:
            config_path (Optional[str]): Path to a saved JSON config file.
        """

        # Detect runtime environment
        env = self.detect_runtime_environment()
        self.is_cluster = env["is_cluster"]
        self.is_grace = env["is_grace"]
        self.prabaha_local = env["prabaha_local"]

        # Set core project directories
        self.project_root = Path(__file__).resolve().parents[3]

        self.config_folder = self.project_root / "src/socialgaze/config/saved_config"
        self.filename = self.get_config_filename()

        # If path was passed in, use it; else, construct default
        if config_path is not None:
            self.config_path = Path(config_path)
        else:
            self.config_path = self.config_folder / self.filename

        self.processed_data_dir = self.project_root / "data/processed"
        self.output_dir = self.project_root / "outputs"
        self.plots_dir = self.output_dir / "plots"

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

        self.file_pattern = "{session_date}_position_{run_number}.mat"

        self.session_names: List[str] = []
        self.runs_by_session: Dict[str, List[str]] = {}

        # Core logic: load or generate config
        if self.config_path.exists():
            logger.info(f"Loading config from {self.config_path}")
            self.load_from_file(self.config_path)
        else:
            logger.info(f"No config found at {self.config_path}. Generating new config.")
            self.initialize_sessions_and_runs()
            ephys_days_and_monkeys_filepath = self.processed_data_dir / "ephys_days_and_monkeys.pkl"
            ephys_days_and_monkeys_df = load_df_from_pkl(ephys_days_and_monkeys_filepath)
            self.extract_sessions_with_ephys_data(ephys_days_and_monkeys_df)
            self.save_to_file(self.config_path)
            logger.info(f"Base config generated and saved to {self.config_path}")

    # -----------------------------
    # Config environment
    # -----------------------------

    def detect_runtime_environment(self) -> dict:
        """Auto-detect runtime environment based on hostname or filesystem."""
        hostname = socket.gethostname().lower()
        username = getpass.getuser()

        if "grace" in hostname or os.path.exists("/gpfs/gibbs/"):
            return {"is_cluster": True, "is_grace": True, "prabaha_local": False}
        elif "milgram" in hostname or os.path.exists("/gpfs/milgram/"):
            return {"is_cluster": True, "is_grace": False, "prabaha_local": False}
        elif username == "prabaha":
            return {"is_cluster": False, "is_grace": False, "prabaha_local": True}
        else:
            return {"is_cluster": False, "is_grace": False, "prabaha_local": False}



    # -----------------------------
    # Session / run discovery
    # -----------------------------

    def get_config_filename(self) -> str:
        if self.is_cluster:
            return "grace_config.json" if self.is_grace else "milgram_config.json"
        return "local_config.json"

    def initialize_sessions_and_runs(self) -> None:
        self.session_names, self.runs_by_session = find_valid_sessions(self,
            path_fns={
                'time': get_time_file_path,
                'pupil': get_pupil_file_path,
                'roi': get_roi_file_path,
            }
        )

    def extract_sessions_with_ephys_data(self, ephys_days_and_monkeys_df: pd.DataFrame) -> None:
        self.session_names, self.runs_by_session = filter_sessions_with_ephys(
            self.session_names,
            self.runs_by_session,
            ephys_days_and_monkeys_df
        )

    # -----------------------------
    # Save / load
    # -----------------------------

    def save_to_file(self, config_path: str) -> None:
        save_config_to_json(object_to_dict(self), config_path)

    def load_from_file(self, config_path: str) -> None:
        config_data = load_config_from_json(config_path)
        assign_dict_attributes_to_object(self, config_data)
