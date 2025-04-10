# src/socialgaze/utils/path_utils.py

from pathlib import Path


def determine_root_data_dir(is_cluster: bool, is_grace: bool, prabaha_local: bool) -> str:
    """
    Determines the root directory for raw data based on the current runtime environment.

    Args:
        is_cluster (bool): Whether the code is running on a cluster.
        is_grace (bool): Whether the code is running on the Grace cluster.
        prabaha_local (bool): Whether the code is running on Prabaha's local machine.

    Returns:
        str: Path to the root raw data directory.
    """
    if is_cluster:
        return (
            "/gpfs/gibbs/project/chang/pg496/data_dir/social_gaze/"
            if is_grace else
            "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/"
        )
    if prabaha_local:
        return "/Users/prabaha/data_dir/social_gaze"
    return "../data/raw"


def get_position_file_path(config, session_date: str, run_number: str) -> Path:
    """
    Constructs the full path to the position .mat file for a given session and run.

    Args:
        config: The BaseConfig object containing path and filename info.
        session_date (str): The session date string.
        run_number (str): The run number string.

    Returns:
        Path: Full path to the position file.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return config.position_dir / filename


def get_time_file_path(config, session_date: str, run_number: str) -> Path:
    """
    Constructs the full path to the time .mat file for a given session and run.

    Args:
        config: The BaseConfig object containing path and filename info.
        session_date (str): The session date string.
        run_number (str): The run number string.

    Returns:
        Path: Full path to the time file.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return config.time_dir / filename


def get_pupil_file_path(config, session_date: str, run_number: str) -> Path:
    """
    Constructs the full path to the pupil .mat file for a given session and run.

    Args:
        config: The BaseConfig object containing path and filename info.
        session_date (str): The session date string.
        run_number (str): The run number string.

    Returns:
        Path: Full path to the pupil file.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return config.pupil_dir / filename


def get_roi_file_path(config, session_date: str, run_number: str) -> Path:
    """
    Constructs the full path to the ROI .mat file for a given session and run.

    Args:
        config: The BaseConfig object containing path and filename info.
        session_date (str): The session date string.
        run_number (str): The run number string.

    Returns:
        Path: Full path to the ROI file.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return config.roi_dir / filename
