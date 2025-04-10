# src/socialgaze/data/behav_data_loader.py

from typing import Optional, Dict, List, Callable, Any
import logging
import pandas as pd
from scipy.io import loadmat

from socialgaze.utils.path_utils import (
    get_position_file_path,
    get_pupil_file_path,
    get_roi_file_path,
    get_time_file_path
)

logger = logging.getLogger(__name__)


# -----------------------------
# .mat file loading utilities
# -----------------------------

def load_mat_from_path(path) -> dict:
    """
    Loads a .mat file and returns its contents as a dictionary.

    Args:
        path: Path to the .mat file.

    Returns:
        dict: Parsed contents of the .mat file.
    """
    return loadmat(str(path), simplify_cells=False)


def _extract_aligned_struct(mat_data: dict):
    """
    Helper function to extract aligned data structure from MATLAB dictionary.

    Args:
        mat_data (dict): Loaded .mat file content.

    Returns:
        np.ndarray or None: The aligned data structure if found.
    """
    for key in ['var', 'aligned_position_file']:
        if key in mat_data:
            return mat_data[key][0][0]
    return None


def extract_roi_dict(agent_roi_data) -> Optional[Dict[str, List[float]]]:
    """
    Converts MATLAB structured ROI data into a Python dictionary.

    Args:
        agent_roi_data: Structured array of ROIs for one agent.

    Returns:
        Optional[Dict[str, List[float]]]: Mapping from ROI name to bounding box coordinates.
    """
    if agent_roi_data is None:
        return None
    return {
        roi_name: agent_roi_data[roi_name][0][0][0]
        for roi_name in agent_roi_data.dtype.names
    }


# -----------------------------
# .mat file processing functions
# -----------------------------

def process_position_file(mat_file, session_name: str, run_number: str, agent: str) -> Optional[pd.DataFrame]:
    """
    Extracts aligned position (x, y) data from a .mat file.

    Returns:
        Optional[pd.DataFrame]: A single-row DataFrame with session/run/agent/x/y or None.
    """
    mat_data = load_mat_from_path(mat_file)
    aligned = _extract_aligned_struct(mat_data)
    if aligned is not None and agent in aligned.dtype.names:
        data = aligned[agent]
        if data is not None and data.size > 0:
            return pd.DataFrame({
                "session_name": [session_name],
                "run_number": [run_number],
                "agent": [agent],
                "x": [data[0, :]],
                "y": [data[1, :]]
            })
    return None


def process_pupil_file(mat_file, session_name: str, run_number: str, agent: str) -> Optional[pd.DataFrame]:
    """
    Extracts aligned pupil size data from a .mat file.

    Returns:
        Optional[pd.DataFrame]: A single-row DataFrame with session/run/agent/pupil_size or None.
    """
    mat_data = load_mat_from_path(mat_file)
    aligned = _extract_aligned_struct(mat_data)
    if aligned is not None and agent in aligned.dtype.names:
        data = aligned[agent]
        if data is not None and data.size > 0:
            return pd.DataFrame({
                "session_name": [session_name],
                "run_number": [run_number],
                "agent": [agent],
                "pupil_size": [data.flatten()]
            })
    return None


def process_time_file(mat_file, session_name: str, run_number: str) -> Optional[pd.DataFrame]:
    """
    Extracts the time vector from a .mat file.

    Returns:
        Optional[pd.DataFrame]: A single-row DataFrame with session/run/neural_timeline or None.
    """
    mat_data = load_mat_from_path(mat_file)
    for key in ['time_file', 'aligned_position_file', 'var']:
        if key in mat_data and 't' in mat_data[key][0][0].dtype.names:
            t = mat_data[key][0][0]['t']
            return pd.DataFrame({
                "session_name": [session_name],
                "run_number": [run_number],
                "neural_timeline": [t.flatten()]
            })
    return None


def process_roi_rects_file(mat_file, session_name: str, run_number: str, agent: str) -> Optional[pd.DataFrame]:
    """
    Extracts ROI bounding boxes for a given agent from a .mat file.

    Returns:
        Optional[pd.DataFrame]: A DataFrame with one row per ROI or None.
    """
    mat_data = load_mat_from_path(mat_file)
    if 'roi_rects' in mat_data:
        roi_data = mat_data['roi_rects'][0][0]
        if agent in roi_data.dtype.names:
            roi_dict = extract_roi_dict(roi_data[agent])
            rows = []
            for roi_name, bbox in roi_dict.items():
                if agent == 'm2' and 'object' in roi_name.lower():
                    continue  # Skip object ROIs for m2
                rows.append({
                    "session_name": session_name,
                    "run_number": run_number,
                    "agent": agent,
                    "roi_name": roi_name,
                    "x_min": bbox[0],
                    "y_min": bbox[1],
                    "x_max": bbox[2],
                    "y_max": bbox[3],
                })
            return pd.DataFrame(rows)
    return None


# -----------------------------
# Data type to function mapping
# -----------------------------

def generate_behav_data_loader_dict(behav_data_types: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Generates a mapping from behavioral data types to their corresponding path and processing functions.

    Args:
        behav_data_types (List[str]): List of behavioral data type names to include.
            Valid values: 'positions', 'pupil', 'roi_vertices', 'neural_timeline'.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary where each key is a behavioral data type and each value is a dict with:
            - "path_func": function to compute the file path
            - "process_func": function to load and process the .mat file
            - "agent_specific": whether the file/data is agent-specific (True/False)
    """
    process_funcs: Dict[str, Callable] = {
        "positions": process_position_file,
        "pupil": process_pupil_file,
        "roi_vertices": process_roi_rects_file,
        "neural_timeline": process_time_file
    }

    path_funcs: Dict[str, Callable] = {
        "positions": get_position_file_path,
        "pupil": get_pupil_file_path,
        "roi_vertices": get_roi_file_path,
        "neural_timeline": get_time_file_path
    }

    agent_specific_flags: Dict[str, bool] = {
        "positions": True,
        "pupil": True,
        "roi_vertices": True,
        "neural_timeline": False
    }

    return {
        dtype: {
            "path_func": path_funcs[dtype],
            "process_func": process_funcs[dtype],
            "agent_specific": agent_specific_flags[dtype]
        }
        for dtype in behav_data_types
    }
