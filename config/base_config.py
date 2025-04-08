# config/base_config.py

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from config.environment import detect_runtime_environment

logger = logging.getLogger(__name__)

class BaseConfig:
    def __init__(self, config_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        env = detect_runtime_environment()
        self.is_cluster = env["is_cluster"]
        self.is_grace = env["is_grace"]
        self.prabaha_local = env["prabaha_local"]
        self.params = params or {}

        self.root_data_dir = self._determine_root_data_dir()
        self.project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        self.data_dir = Path(self.root_data_dir) / "raw"
        self.processed_dir = self.project_root / "processed"
        self.output_dir = self.project_root / "outputs"
        self.plots_dir = self.output_dir / "plots"

        self.position_dir = self.data_dir / "eyetracking/aligned_raw_samples/position"
        self.time_dir = self.data_dir / "eyetracking/aligned_raw_samples/time"
        self.pupil_dir = self.data_dir / "eyetracking/aligned_raw_samples/pupil_size"
        self.roi_dir = self.data_dir / "eyetracking/roi_rects"

        self.file_pattern = "{session_date}_position_{run_number}.mat"

        self.session_dates: List[str] = []
        self.runs_by_session: Dict[str, List[str]] = {}

        if config_path:
            self.load_from_file(config_path)

        self.initialize_sessions_and_runs()

    # -----------------------------
    # Path logic
    # -----------------------------

    def _determine_root_data_dir(self) -> str:
        if self.is_cluster:
            return "/gpfs/gibbs/project/chang/pg496/data_dir/social_gaze/" if self.is_grace \
                else "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/"
        elif self.prabaha_local:
            return "/Users/prabaha/data_dir/social_gaze"
        else:
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), "social_gaze")

    def get_position_file_path(self, session_date: str, run_number: str) -> Path:
        filename = self.file_pattern.format(session_date=session_date, run_number=run_number)
        return self.position_dir / filename

    def get_time_file_path(self, session_date: str, run_number: str) -> Path:
        filename = self.file_pattern.format(session_date=session_date, run_number=run_number)
        return self.time_dir / filename

    def get_pupil_file_path(self, session_date: str, run_number: str) -> Path:
        filename = self.file_pattern.format(session_date=session_date, run_number=run_number)
        return self.pupil_dir / filename

    def get_roi_file_path(self, session_date: str, run_number: str) -> Path:
        filename = self.file_pattern.format(session_date=session_date, run_number=run_number)
        return self.roi_dir / filename

    def get_processed_path(self, session_date: str, run_number: str, data_type: str, extension: str = 'csv') -> Path:
        filename = f"{data_type}_{session_date}_run_{run_number}.{extension}"
        return self.processed_dir / filename

    # -----------------------------
    # Session / run discovery
    # -----------------------------

    def initialize_sessions_and_runs(self) -> None:
        """
        Automatically discovers session dates and run numbers based on the presence of
        .mat files in the expected folders. Only includes sessions where all 4 types of
        data (position, pupil, time, ROI) exist.
        """
        position_files = sorted(self.position_dir.glob("*.mat"))
        session_run_pattern = re.compile(r"(\d{7,8})_position_(\d+)\.mat")

        self.session_dates = []
        self.runs_by_session = {}

        for file in position_files:
            match = session_run_pattern.match(file.name)
            if not match:
                continue
            session_date, run_number = match.groups()

            # Check for the other required files
            required_files = [
                self.get_time_file_path(session_date, run_number),
                self.get_pupil_file_path(session_date, run_number),
                self.get_roi_file_path(session_date, run_number),
            ]
            if all(f.exists() for f in required_files):
                if session_date not in self.session_dates:
                    self.session_dates.append(session_date)
                    self.runs_by_session[session_date] = []
                self.runs_by_session[session_date].append(run_number)
            else:
                missing = [f.name for f in required_files if not f.exists()]
                logger.warning(f"Skipping session {session_date}, run {run_number} — missing files: {missing}")

    # -----------------------------
    # Save / load
    # -----------------------------

    def save_to_file(self, config_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def load_from_file(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
