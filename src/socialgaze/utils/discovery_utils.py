# src/socialgaze/utils/discovery_utils.py


import re
from typing import List, Dict, Tuple, Callable, Any
from pathlib import Path
import logging

import pdb

logger = logging.getLogger(__name__)


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
            logger.info(f"Discarding session {session_name} — no ephys data found.")

    return filtered_session_names, filtered_runs_by_session
