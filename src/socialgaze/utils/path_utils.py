# src/socialgaze/utils/path_utils.py


from pathlib import Path
import pdb
from socialgaze.config.base_config import BaseConfig


def get_position_file_path(config: BaseConfig, session_date: str, run_number: str) -> Path:
    """
    Returns the full path to the position .mat file for a given session and run.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return Path(config.position_dir) / filename


def get_time_file_path(config: BaseConfig, session_date: str, run_number: str) -> Path:
    """
    Returns the full path to the time .mat file for a given session and run.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return Path(config.time_dir) / filename


def get_pupil_file_path(config: BaseConfig, session_date: str, run_number: str) -> Path:
    """
    Returns the full path to the pupil .mat file for a given session and run.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return Path(config.pupil_dir) / filename


def get_roi_file_path(config: BaseConfig, session_date: str, run_number: str) -> Path:
    """
    Returns the full path to the ROI .mat file for a given session and run.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return Path(config.roi_dir) / filename
