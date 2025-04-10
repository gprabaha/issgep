# src/socialgaze/utils/path_utils.py


from pathlib import Path
import pdb


def determine_root_data_dir(is_cluster: bool, is_grace: bool, prabaha_local: bool) -> str:
    if is_cluster:
        return "/gpfs/gibbs/project/chang/pg496/data_dir/social_gaze/" if is_grace \
            else "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/"
    if prabaha_local:
        return "/Users/prabaha/data_dir/social_gaze"
    return "../data/raw"


def get_position_file_path(config, session_date: str, run_number: str) -> Path:
    """
    Returns the full path to the position .mat file for a given session and run.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return Path(config.position_dir) / filename


def get_time_file_path(config, session_date: str, run_number: str) -> Path:
    """
    Returns the full path to the time .mat file for a given session and run.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return Path(config.time_dir) / filename


def get_pupil_file_path(config, session_date: str, run_number: str) -> Path:
    """
    Returns the full path to the pupil .mat file for a given session and run.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return Path(config.pupil_dir) / filename


def get_roi_file_path(config, session_date: str, run_number: str) -> Path:
    """
    Returns the full path to the ROI .mat file for a given session and run.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return Path(config.roi_dir) / filename
