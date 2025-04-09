# src/socialgaze/config/base_config.py

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from socialgaze.config.environment import detect_runtime_environment
from socialgaze.data.extract_data_from_mat_files import (
    process_position_file,
    process_time_file,
    process_pupil_file,
    process_roi_rects_file,
)

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
        
        self.behav_data_registry = {
            "positions": {
                "path_func": self.get_position_file_path,
                "process_func": process_position_file,
                "agent_specific": True
            },
            "pupil": {
                "path_func": self.get_pupil_file_path,
                "process_func": process_pupil_file,
                "agent_specific": True
            },
            "roi_vertices": {
                "path_func": self.get_roi_file_path,
                "process_func": process_roi_rects_file,
                "agent_specific": True
            },
            "neural_timeline": {
                "path_func": self.get_time_file_path,
                "process_func": process_time_file,
                "agent_specific": False
            }
        }

        self.data_dir = Path(self.determine_root_data_dir())
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

    def get_position_file_path(self, session_date: str, run_number: str) -> Path:
        """
        Returns the full path to the position .mat file for a given session and run.

        Args:
            session_date (str): The date of the session.
            run_number (str): The run number.

        Returns:
            Path: Path to the position file.
        """
        filename = self.file_pattern.format(session_date=session_date, run_number=run_number)
        return self.position_dir / filename

    def get_time_file_path(self, session_date: str, run_number: str) -> Path:
        """Returns the full path to the time .mat file."""
        filename = self.file_pattern.format(session_date=session_date, run_number=run_number)
        return self.time_dir / filename

    def get_pupil_file_path(self, session_date: str, run_number: str) -> Path:
        """Returns the full path to the pupil .mat file."""
        filename = self.file_pattern.format(session_date=session_date, run_number=run_number)
        return self.pupil_dir / filename

    def get_roi_file_path(self, session_date: str, run_number: str) -> Path:
        """Returns the full path to the ROI .mat file."""
        filename = self.file_pattern.format(session_date=session_date, run_number=run_number)
        return self.roi_dir / filename

    def get_processed_path(self, session_date: str, run_number: str, data_type: str, extension: str = 'csv') -> Path:
        """
        Returns the path for saving a processed file for a given session and run.

        Args:
            session_date (str): The session date.
            run_number (str): The run number.
            data_type (str): Type of data (e.g., 'positions', 'pupil').
            extension (str): File extension (default is 'csv').

        Returns:
            Path: Full path to the output file.
        """
        filename = f"{data_type}_{session_date}_run_{run_number}.{extension}"
        return self.processed_data_dir / filename

    # -----------------------------
    # Session / run discovery
    # -----------------------------

    def initialize_sessions_and_runs(self) -> None:
        """
        Scans the position directory and automatically detects available session dates
        and their corresponding run numbers. A session/run pair is only included if
        all required .mat files (time, pupil, ROI) are found.
        """
        position_files = sorted(self.position_dir.glob("*.mat"))
        session_run_pattern = re.compile(r"(\d{7,8})_position_(\d+)\.mat")

        session_names = []
        runs_by_session = {}

        for file in position_files:
            match = session_run_pattern.match(file.name)
            if not match:
                continue
            session_name, run_number = match.groups()

            required_files = [
                self.get_time_file_path(session_name, run_number),
                self.get_pupil_file_path(session_name, run_number),
                self.get_roi_file_path(session_name, run_number),
            ]
            if all(f.exists() for f in required_files):
                if session_name not in self.session_names:
                    session_names.append(session_name)
                    runs_by_session[session_name] = []
                runs_by_session[session_name].append(run_number)
            else:
                missing = [f.name for f in required_files if not f.exists()]
                logger.warning(f"Skipping session {session_name}, run {run_number} — missing files: {missing}")
            self.session_names = session_names
            self.runs_by_session = runs_by_session

    # -----------------------------
    # Save / load
    # -----------------------------

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
