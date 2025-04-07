# config/base_config.py
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from config.environment import detect_runtime_environment


class BaseConfig:
    def __init__(self, config_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        self.params = params or detect_runtime_environment()
        self.root_data_dir = self._determine_root_data_dir()
        self.project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        self.data_dir = Path(self.root_data_dir) / "raw"
        self.processed_dir = Path(self.root_data_dir) / "processed"
        self.output_dir = self.project_root / "outputs"
        self.plots_dir = self.output_dir / "plots"

        self.position_dir = self.data_dir / "eyetracking/aligned_raw_samples/position"
        self.time_dir = self.data_dir / "eyetracking/aligned_raw_samples/time"
        self.pupil_dir = self.data_dir / "eyetracking/aligned_raw_samples/pupil_size"
        self.roi_dir = self.data_dir / "eyetracking/roi_rects"

        self.file_pattern = "{session_date}_position_{run_number}.mat"
        self.session_dates = []
        self.runs_by_session = {}

        if config_path:
            self.load_from_file(config_path)

    def _determine_root_data_dir(self) -> str:
        if self.params.get('is_cluster', False):
            return "/gpfs/gibbs/project/chang/pg496/data_dir/social_gaze/" if self.params.get('is_grace', False) \
                else "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/"
        elif self.params.get('prabaha_local', False):
            return "/Users/prabaha/data_dir/social_gaze"
        else:
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), "social_gaze")

    def load_from_file(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_to_file(self, config_path: str) -> None:
        config_dict = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in self.__dict__.items() if not key.startswith('_')
        }
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

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

    def add_session(self, session_date: str, run_numbers: List[str]) -> None:
        if session_date not in self.session_dates:
            self.session_dates.append(session_date)
        self.runs_by_session[session_date] = run_numbers
