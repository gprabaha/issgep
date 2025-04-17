# src/socialgaze/utils/path_utils.py

from pathlib import Path
from typing import Dict, Optional, List

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


# Config

def get_project_root() -> Path:
    """
    Returns the root directory of the project.

    This function assumes it is located 3 directories deep inside the project
    (e.g., inside src/socialgaze/utils/) and moves upward to find the root.

    Returns:
        Path: The root path of the entire project.
    """
    return Path(__file__).resolve().parents[3]


def get_default_config_folder(project_root: Path) -> Path:
    """
    Returns the default configuration folder path and ensures it exists.

    This folder is typically used to store JSON config files generated during runtime.

    Args:
        project_root (Path): The root directory of the project.

    Returns:
        Path: Path to the config folder (e.g., src/socialgaze/config/saved_configs).
    """
    config_folder = project_root / "src/socialgaze/config/saved_configs"
    config_folder.mkdir(parents=True, exist_ok=True)
    return config_folder


def get_default_data_paths(project_root: Path) -> Dict[str, Path]:
    """
    Constructs default paths to processed data, output, and plot folders.

    Args:
        project_root (Path): The root directory of the project.

    Returns:
        Dict[str, Path]: A dictionary with keys 'processed', 'outputs', and 'plots'.
    """
    return {
        "processed": project_root / "data/processed",
        "outputs": project_root / "outputs",
        "plots": project_root / "outputs/plots",
    }


def get_raw_data_directories(data_root: Path) -> Dict[str, Path]:
    """
    Returns standardized subdirectories for raw eyetracking data under the given root.

    Args:
        data_root (Path): The root directory for raw data (e.g., /gpfs/.../data)

    Returns:
        Dict[str, Path]: Dictionary containing paths for 'position', 'time', 'pupil', and 'roi'
    """
    return {
        "position": data_root / "eyetracking/aligned_raw_samples/position",
        "time": data_root / "eyetracking/aligned_raw_samples/time",
        "pupil": data_root / "eyetracking/aligned_raw_samples/pupil_size",
        "roi": data_root / "eyetracking/roi_rects",
    }


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
