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
    get_project_root,
    get_default_config_folder,
    get_default_data_paths,
    get_raw_data_directories,
    get_pupil_file_path,
    get_roi_file_path,
    get_time_file_path,
    join_folder_and_filename
)
from socialgaze.utils.discovery_utils import (
    get_config_filename,
    get_mat_filename_pattern,
    find_valid_sessions,
    filter_sessions_with_ephys
)
from socialgaze.utils.saving_utils import save_config_to_json
from socialgaze.utils.conversion_utils import object_to_dict, assign_dict_attributes_to_object

import pdb

logger = logging.getLogger(__name__)


class BaseConfig:
    """
    A configuration manager for handling paths, environment flags,
    and available session/run combinations.

    Responsibilities:
    - Detect runtime environment (cluster/local)
    - Set default paths (raw, processed, outputs, config)
    - Determine valid sessions and runs
    - Save/load config from a JSON file
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the BaseConfig object.

        If a saved config exists at the provided path, it loads it.
        Otherwise, it auto-detects the environment, sets up paths, and scans for valid sessions/runs.
        On non-cluster environments, raw data paths and scanning are skipped.

        Args:
            config_path (Optional[str]): Optional path to an existing saved config file.
        """
        self.detect_runtime_environment()

        self.project_root = get_project_root()
        self.config_folder = get_default_config_folder(self.project_root)
        self.filename = get_config_filename(self.is_cluster, self.is_grace)

        # Determine config save path
        self.config_path = Path(config_path) if config_path else join_folder_and_filename(self.config_folder, self.filename)

        # Core processed output paths (these are safe to set in any environment)
        paths = get_default_data_paths(self.project_root)
        self.processed_data_dir = paths["processed"]
        self.output_dir = paths["outputs"]
        self.plots_dir = paths["plots"]

        self.behav_data_types = ['positions', 'roi_vertices', 'pupil', 'neural_timeline']
        self.session_names: List[str] = []
        self.runs_by_session: Dict[str, List[str]] = {}

        # Main logic: Load existing config or generate new one
        if self.config_path.exists():
            logger.info(f"Loading existing config from {self.config_path}")
            self.load_from_file(self.config_path)
        else:
            logger.info(f"No config found at {self.config_path}. Generating new config.")

            if self.is_cluster:
                # Only define and validate raw data paths if on a cluster
                self.data_dir = determine_root_data_dir(
                    is_cluster=self.is_cluster,
                    is_grace=self.is_grace,
                    prabaha_local=self.prabaha_local
                )
                raw_dirs = get_raw_data_directories(self.data_dir)
                self.position_dir = raw_dirs["position"]
                self.time_dir = raw_dirs["time"]
                self.pupil_dir = raw_dirs["pupil"]
                self.roi_dir = raw_dirs["roi"]

                self.file_pattern = get_mat_filename_pattern()
                self.initialize_sessions_and_runs()

                ephys_days_and_monkeys_filepath = self.processed_data_dir / "ephys_days_and_monkeys.pkl"
                ephys_days_and_monkeys_df = load_df_from_pkl(ephys_days_and_monkeys_filepath)

                self.extract_sessions_with_ephys_data(ephys_days_and_monkeys_df)

            self.save_to_file(self.config_path)
            logger.info(f"Base config generated and saved to {self.config_path}")


    # -----------------------------
    # Config environment
    # -----------------------------

    def detect_runtime_environment(self) -> None:
        """
        Auto-detects the machine/runtime environment and sets flags:
        - is_cluster: whether running on HPC
        - is_grace: whether running on Grace cluster
        - prabaha_local: whether running locally on Prabaha's machine
        """
        hostname = socket.gethostname().lower()
        username = getpass.getuser()

        if "grace" in hostname or os.path.exists("/gpfs/gibbs/"):
            self.is_cluster = True
            self.is_grace = True
            self.prabaha_local = False
        elif "milgram" in hostname or os.path.exists("/gpfs/milgram/"):
            self.is_cluster = True
            self.is_grace = False
            self.prabaha_local = False
        elif username == "prabaha":
            self.is_cluster = False
            self.is_grace = False
            self.prabaha_local = True
        else:
            self.is_cluster = False
            self.is_grace = False
            self.prabaha_local = False

    # -----------------------------
    # Session / run discovery
    # -----------------------------

    def initialize_sessions_and_runs(self) -> None:
        """
        Populates `self.session_names` and `self.runs_by_session`
        using a validity check on required .mat files.
        """
        self.session_names, self.runs_by_session = find_valid_sessions(
            self,
            path_fns={
                'time': get_time_file_path,
                'pupil': get_pupil_file_path,
                'roi': get_roi_file_path,
            }
        )

    def extract_sessions_with_ephys_data(self, ephys_days_and_monkeys_df: pd.DataFrame) -> None:
        """
        Filters session/run list to only include those with matching ephys data.

        Args:
            ephys_days_and_monkeys_df (pd.DataFrame): Contains ephys recording metadata.
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
        Serializes the config as a dictionary and saves to a JSON file.

        Args:
            config_path (str): Path where the config should be saved.
        """
        save_config_to_json(object_to_dict(self), config_path)

    def load_from_file(self, config_path: str) -> None:
        """
        Loads config values from a JSON file and sets them as instance attributes.

        Args:
            config_path (str): Path to the saved config file.
        """
        config_data = load_config_from_json(config_path)
        assign_dict_attributes_to_object(self, config_data)


