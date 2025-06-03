# src/socialgaze/utils/discovery_utils.py

import re
import os
from typing import List, Dict, Tuple, Callable, Any
from pathlib import Path
from multiprocessing import cpu_count
import logging

logger = logging.getLogger(__name__)


def get_config_filename(is_cluster: bool, is_grace: bool) -> str:
    """
    Returns the default config filename based on environment flags.

    Args:
        is_cluster (bool): Whether the code is running on a cluster.
        is_grace (bool): Whether the environment is Grace within the cluster.

    Returns:
        str: Name of the config file (e.g., "milgram_config.json").
    """
    if is_cluster:
        return "grace_config.json" if is_grace else "milgram_config.json"
    return "local_config.json"


def get_mat_filename_pattern():
    """
    Returns the standardized filename pattern used for .mat files
    containing aligned raw gaze position data.

    The returned string can be formatted with `session_date` and `run_number`
    to generate full filenames (e.g., "20240315_position_3.mat").

    Returns:
        str: Filename pattern string with placeholders.
    """
    return "{session_date}_position_{run_number}.mat"


def find_valid_sessions(
    config: Any,
    path_fns: Dict[str, Callable[[str, str], Path]]
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Scans the position directory for session/run combinations and validates that
    required files (e.g., time, pupil, ROI) exist for each pair.

    Args:
        position_dir (Path): Path to the directory containing position files.
        file_pattern (str): Regex pattern to extract session_name and run_number.
        path_fns (Dict[str, Callable[[str, str], Path]]): Dict of required path functions.

    Returns:
        Tuple[List[str], Dict[str, List[str]]]: Valid session names and their corresponding runs.
    """
    position_files = sorted(config.position_dir.glob("*.mat"))
    session_run_pattern = re.compile(r"(\d{7,8})_position_(\d+)\.mat")

    session_names: List[str] = []
    runs_by_session: Dict[str, List[str]] = {}

    for file in position_files:
        match = session_run_pattern.match(file.name)
        if not match:
            continue

        session_name, run_number = match.groups()

        required_paths = [path_fn(config, session_name, run_number) for path_fn in path_fns.values()]
        if all(f.exists() for f in required_paths):
            if session_name not in session_names:
                session_names.append(session_name)
                runs_by_session[session_name] = []
            runs_by_session[session_name].append(run_number)
        else:
            missing = [f.name for f in required_paths if not f.exists()]
            logger.warning(f"Skipping session {session_name}, run {run_number} — missing files: {missing}")

    return session_names, runs_by_session


def filter_sessions_with_ephys(
    session_names: List[str],
    runs_by_session: Dict[str, List[str]],
    ephys_days_df
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Filters out sessions that do not have ephys data according to the dataframe.

    Args:
        session_names (List[str]): All detected sessions.
        runs_by_session (Dict[str, List[str]]): Mapping of session names to run numbers.
        ephys_days_df (pd.DataFrame): DataFrame with a 'session_name' column listing valid sessions.

    Returns:
        Tuple[List[str], Dict[str, List[str]]]: Filtered session names and run mapping.
    """
    valid_ephys_sessions = set(ephys_days_df['session_name'].astype(str))
    filtered_session_names = []
    filtered_runs_by_session = {}

    for session_name in session_names:
        if session_name in valid_ephys_sessions:
            filtered_session_names.append(session_name)
            filtered_runs_by_session[session_name] = runs_by_session[session_name]
        else:
            # logger.warning(f"Discarding session {session_name} — no ephys data found.")

    return filtered_session_names, filtered_runs_by_session

def get_num_available_cpus(is_cluster: bool) -> int:
    """Returns the number of CPUs available for parallel processing."""
    if is_cluster:
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
        return int(slurm_cpus) if slurm_cpus else 1
    else:
        return cpu_count()
