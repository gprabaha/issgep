# src/data/extract_data_from_mat_files.py


from typing import Optional, Dict
import logging
import pandas as pd
from scipy.io import loadmat

import pdb


logger = logging.getLogger(__name__)


def load_mat_from_path(path):
    return loadmat(str(path), simplify_cells=False)


def process_position_file(mat_file, session_name: str, run_number: str, agent: str) -> Optional[pd.DataFrame]:
    mat_data = load_mat_from_path(mat_file)
    aligned = None
    if 'var' in mat_data:
        aligned = mat_data['var'][0][0]
    elif 'aligned_position_file' in mat_data:
        aligned = mat_data['aligned_position_file'][0][0]
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
    mat_data = load_mat_from_path(mat_file)
    aligned = None
    if 'var' in mat_data:
        aligned = mat_data['var'][0][0]
    elif 'aligned_position_file' in mat_data:
        aligned = mat_data['aligned_position_file'][0][0]
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
    mat_data = load_mat_from_path(mat_file)
    if 'roi_rects' in mat_data:
        roi_data = mat_data['roi_rects'][0][0]
        if agent in roi_data.dtype.names:
            roi_dict = extract_roi_dict(roi_data[agent])
            rows = []
            for roi_name, bbox in roi_dict.items():
                if agent == 'm2' and 'object' in roi_name.lower():
                    continue
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


def extract_roi_dict(agent_roi_data) -> Optional[Dict]:
    if agent_roi_data is None:
        return None
    return {
        roi_name: agent_roi_data[roi_name][0][0][0]
        for roi_name in agent_roi_data.dtype.names
    }
