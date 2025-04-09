# src/socialgaze/utils/preprocessing_utils.py


import logging

from socialgaze.utils.path_utils import (
    get_position_file_path,
    get_pupil_file_path,
    get_roi_file_path,
    get_time_file_path
)
from socialgaze.data.extract_data_from_mat_files import (
    process_position_file,
    process_time_file,
    process_pupil_file,
    process_roi_rects_file
)


logger = logging.getLogger(__name__)


def generate_behav_data_loader_registry(behav_data_types):
    process_funcs = {
        "positions": process_position_file,
        "pupil": process_pupil_file,
        "roi_vertices": process_roi_rects_file,
        "neural_timeline": process_time_file
    }
    path_funcs = {
        "positions": get_position_file_path,
        "pupil": get_pupil_file_path,
        "roi_vertices": get_roi_file_path,
        "neural_timeline": get_time_file_path
    }
    agent_specific_flags = {
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
