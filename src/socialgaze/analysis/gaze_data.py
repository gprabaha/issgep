import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.path_utils import (
    get_position_file_path,
    get_pupil_file_path,
    get_roi_file_path,
    get_time_file_path
)

logger = logging.getLogger(__name__)


class GazeData:
    def __init__(self, config: BaseConfig):
        """
        Initializes the GazeData loader with the provided configuration.
        """
        self.config = config
        self.behav_data_loader_dict = self._generate_behav_data_loader_dict(config.behav_data_types)
        self.timeseries: Dict[str, List[pd.DataFrame]] = defaultdict(list)

    # -----------------------------
    # Internal utility methods
    # -----------------------------

    def _load_mat_from_path(self, path) -> dict:
        from scipy.io import loadmat
        return loadmat(str(path), simplify_cells=False)

    def _extract_aligned_struct(self, mat_data: dict):
        for key in ['var', 'aligned_position_file']:
            if key in mat_data:
                return mat_data[key][0][0]
        return None

    def _extract_roi_dict(self, agent_roi_data) -> Optional[Dict[str, List[float]]]:
        if agent_roi_data is None:
            return None
        return {
            roi_name: agent_roi_data[roi_name][0][0][0]
            for roi_name in agent_roi_data.dtype.names
        }

    # -----------------------------
    # Internal .mat file processing
    # -----------------------------

    def _process_position_file(self, mat_file, session_name: str, run_number: str, agent: str) -> Optional[pd.DataFrame]:
        mat_data = self._load_mat_from_path(mat_file)
        aligned = self._extract_aligned_struct(mat_data)
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

    def _process_pupil_file(self, mat_file, session_name: str, run_number: str, agent: str) -> Optional[pd.DataFrame]:
        mat_data = self._load_mat_from_path(mat_file)
        aligned = self._extract_aligned_struct(mat_data)
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

    def _process_time_file(self, mat_file, session_name: str, run_number: str) -> Optional[pd.DataFrame]:
        mat_data = self._load_mat_from_path(mat_file)
        for key in ['time_file', 'aligned_position_file', 'var']:
            if key in mat_data and 't' in mat_data[key][0][0].dtype.names:
                t = mat_data[key][0][0]['t']
                return pd.DataFrame({
                    "session_name": [session_name],
                    "run_number": [run_number],
                    "neural_timeline": [t.flatten()]
                })
        return None

    def _process_roi_rects_file(self, mat_file, session_name: str, run_number: str, agent: str) -> Optional[pd.DataFrame]:
        mat_data = self._load_mat_from_path(mat_file)
        if 'roi_rects' in mat_data:
            roi_data = mat_data['roi_rects'][0][0]
            if agent in roi_data.dtype.names:
                roi_dict = self._extract_roi_dict(roi_data[agent])
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
    # Internal loader registry
    # -----------------------------

    def _generate_behav_data_loader_dict(self, behav_data_types: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Constructs mapping from behavioral data types to their loaders.
        """

        path_funcs: Dict[str, Callable] = {
            "positions": get_position_file_path,
            "pupil": get_pupil_file_path,
            "roi_vertices": get_roi_file_path,
            "neural_timeline": get_time_file_path
        }

        process_funcs: Dict[str, Callable] = {
            "positions": self._process_position_file,
            "pupil": self._process_pupil_file,
            "roi_vertices": self._process_roi_rects_file,
            "neural_timeline": self._process_time_file
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


    def load_raw_data_from_mat_files(self,
                    session_filter: Optional[Union[str, List[str]]] = None,
                    run_filter: Optional[Union[str, List[str]]] = None,
                    agent_filter: Optional[Union[str, List[str]]] = None,
                    data_types: Optional[List[str]] = None):
        """
        Loads behavioral data from .mat files and populates self.timeseries.

        Args:
            session_filter, run_filter, agent_filter: Optional filters.
            data_types: List of data types to load.
        """
        active_data_types = data_types if data_types else list(self.behav_data_loader_dict.keys())

        for data_type in active_data_types:
            registry_entry = self.behav_data_loader_dict[data_type]
            path_func = registry_entry["path_func"]
            process_func = registry_entry["process_func"]
            agent_specific = registry_entry["agent_specific"]

            for session_name in self.config.session_names:
                if session_filter and session_name not in session_filter:
                    continue

                for run_number in self.config.runs_by_session.get(session_name, []):
                    run_number = str(run_number)
                    if run_filter and run_number not in run_filter:
                        continue

                    if agent_specific:
                        agents = agent_filter if agent_filter else ["m1", "m2"]
                        for agent in agents:
                            path = path_func(self.config, session_name, run_number)
                            df = process_func(path, session_name, run_number, agent)
                            if df is not None:
                                self.timeseries[data_type].append(df)
                    else:
                        path = path_func(self.config, session_name, run_number)
                        df = process_func(path, session_name, run_number)
                        if df is not None:
                            self.timeseries[data_type].append(df)
            self.timeseries[data_type] = pd.concat(self.timeseries[data_type], ignore_index=True)


    def load_from_saved_dataframes(self, data_types: Optional[List[str]] = None):
        """
        Loads existing .pkl files into memory and caches them in self.timeseries.

        Args:
            data_types: Only load these data types if provided.
        """
        data_types = data_types if data_types else self.behav_data_loader_dict.keys()
        for data_type in data_types:
            pkl_path = self.config.processed_data_dir / f"{data_type}.pkl"
            if pkl_path.exists():
                try:
                    df = pd.read_pickle(pkl_path)
                    self.timeseries[data_type] = [df]
                    logger.info(f"Loaded {data_type} from {pkl_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {data_type} from disk: {e}")


    def save_as_dataframes(self, output_dir: Path):
        """
        Saves the collected dataframes to .pkl files in the specified directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Combining all dataframes and saving")
        for data_type, df in self.timeseries.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                out_path = output_dir / f"{data_type}.pkl"
                df.to_pickle(out_path)
                logger.info(f"Saved {data_type}.pkl to {out_path}")
        logger.info("All raw data saved as DataFrames.")


    def get_data(self,
                 data_type: str,
                 session_name: Optional[str] = None,
                 run_number: Optional[str] = None,
                 agent: Optional[str] = None,
                 load_if_available: bool = True,
                 fallback_to_mat: bool = True) -> Optional[pd.DataFrame]:
        """
        Retrieves a behavioral data type, optionally filtered.

        Args:
            data_type (str): e.g., 'positions', 'pupil', etc.
            session_name, run_number, agent: Optional filters.
            load_if_available (bool): Try to load from .pkl if not already in memory.
            fallback_to_mat (bool): If not found in memory or .pkl, try loading from raw .mat files.

        Returns:
            Optional[pd.DataFrame]: Filtered DataFrame or None.
        """
        # Try to load from disk if not already cached
        if data_type not in self.timeseries and load_if_available:
            self.load_from_saved_dataframes([data_type])

        # Try fallback to mat files (expensive)
        if data_type not in self.timeseries and fallback_to_mat:
            logger.warning(f"{data_type} not loaded. Attempting raw mat load...")
            logger.info(f"Loading {data_type} from raw .mat files for session={session_name}, run={run_number}, agent={agent}")
            self.load_raw_data_from_mat_files(data_types=[data_type],
                             session_filter=session_name,
                             run_filter=run_number,
                             agent_filter=agent if agent else None)

        if data_type not in self.timeseries or not self.timeseries[data_type]:
            logger.warning(f"{data_type} not available in memory.")
            return None

        df = self.timeseries[data_type]
        if session_name is not None:
            df = df[df["session_name"] == session_name]
        if run_number is not None:
            df = df[df["run_number"] == run_number]
        if agent is not None and "agent" in df.columns:
            df = df[df["agent"] == agent]

        return df if not df.empty else None
