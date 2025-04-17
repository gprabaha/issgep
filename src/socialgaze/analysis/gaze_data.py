# src/socialgaze/analysis/gaze_data.py

import os
import logging
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.path_utils import (
    get_position_file_path,
    get_pupil_file_path,
    get_roi_file_path,
    get_time_file_path
)
from socialgaze.utils.saving_utils import save_df_to_pkl
from socialgaze.utils.loading_utils import load_df_from_pkl, load_mat_from_path

logger = logging.getLogger(__name__)


class GazeData:

    """
    GazeData manages the loading, processing, and access of behavioral gaze-related data.

    Features:
    - Loads raw .mat files for positions, pupil sizes, ROI vertices, and neural timelines.
    - Saves and loads preprocessed data using pickle (.pkl) files.
    - Provides session/run/agent-level filtered access via `get_data`.
    - Performs timeline pruning and NaN interpolation for clean downstream usage.
    """

    # -----------------------------
    # 1. Init & setup
    # -----------------------------

    def __init__(self, config: BaseConfig):
        """
        Initializes the GazeData loader with the provided configuration object.

        Args:
            config (BaseConfig): Contains session/run list and data path settings.
        """
        self.config = config
        self.behav_data_loader_dict = self._generate_behav_data_loader_dict(config.behav_data_types)
        self.raw_data: Dict[str, List[pd.DataFrame]] = defaultdict(list)


    def _generate_behav_data_loader_dict(self, behav_data_types: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Creates a registry mapping each behavioral data type to its path and processing functions.

        Args:
            behav_data_types (List[str]): Data types to include (e.g., 'positions', 'pupil').

        Returns:
            Dict[str, Dict]: Dictionary containing path_func, process_func, and agent_specific flag for each type.
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

    # -----------------------------
    # 2. Public interfaces
    # -----------------------------

    def load_raw_data_from_mat_files(self,
                    session_filter: Optional[Union[str, List[str]]] = None,
                    run_filter: Optional[Union[str, List[str]]] = None,
                    agent_filter: Optional[Union[str, List[str]]] = None,
                    data_types: Optional[List[str]] = None):
        """
        Loads behavioral data from .mat files and stores them in self.raw_data.

        Optional filters can limit loading to specific sessions, runs, or agents.

        Args:
            session_filter (str or List[str], optional): Sessions to load.
            run_filter (str or List[str], optional): Runs to load.
            agent_filter (str or List[str], optional): Agents to load ('m1', 'm2').
            data_types (List[str], optional): Which data types to load.
        """
        active_data_types = data_types if data_types else list(self.behav_data_loader_dict.keys())

        for data_type in active_data_types:
            registry_entry = self.behav_data_loader_dict[data_type]
            path_func = registry_entry["path_func"]
            process_func = registry_entry["process_func"]
            agent_specific = registry_entry["agent_specific"]
            logger.info(f"Loading .mat data for: {data_type}")
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
                                self.raw_data[data_type].append(df)
                    else:
                        path = path_func(self.config, session_name, run_number)
                        df = process_func(path, session_name, run_number)
                        if df is not None:
                            self.raw_data[data_type].append(df)
            self.raw_data[data_type] = pd.concat(self.raw_data[data_type], ignore_index=True)

    # Load and save generated dataframes
    
    def load_from_saved_dataframes(self, data_types: Optional[List[str]] = None):
        """
        Loads previously saved DataFrames from .pkl files into self.raw_data.

        Args:
            data_types (List[str], optional): Data types to load. If None, all are loaded.
        """
        data_types = data_types if data_types else self.behav_data_loader_dict.keys()
        for data_type in data_types:
            pkl_path = self.config.processed_data_dir / f"{data_type}.pkl"
            if pkl_path.exists():
                try:
                    df = load_df_from_pkl(pkl_path)
                    self.raw_data[data_type] = [df]
                    logger.info(f"Loaded {data_type} from {pkl_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {data_type} from disk: {e}")


    def save_as_dataframes(self, output_dir: Optional[Path] = None):
        """
        Saves all in-memory DataFrames in self.raw_data to the specified output directory as .pkl files.

        Args:
            output_dir (Optional[Path]): Directory to save the pickle files in. Defaults to self.config.processed_data_dir.
        """
        output_dir = output_dir or self.config.processed_data_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving raw data dataframes")
        for data_type, df in self.raw_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                out_path = output_dir / f"{data_type}.pkl"
                save_df_to_pkl(df, out_path)
        logger.info("All raw data saved as DataFrames.")


    # Fetch any part of the data

    def get_data(self,
                data_type: str,
                session_name: Optional[str] = None,
                run_number: Optional[str] = None,
                agent: Optional[str] = None,
                load_if_available: bool = True,
                fallback_to_mat: bool = True) -> Optional[pd.DataFrame]:
        """
        Retrieves a behavioral data type (e.g., 'positions'), with optional filtering.

        Automatically loads from .pkl or raw .mat files if the data is not yet in memory.

        Args:
            data_type (str): Type of data to fetch.
            session_name (str, optional): Filter by session.
            run_number (str, optional): Filter by run.
            agent (str, optional): Filter by agent ('m1' or 'm2').
            load_if_available (bool): If True, attempt to load from .pkl.
            fallback_to_mat (bool): If True, fallback to loading raw .mat files if needed.

        Returns:
            Optional[pd.DataFrame]: Filtered DataFrame, or None if not available.
        """
        # Check if already loaded
        if data_type not in self.raw_data or self.raw_data[data_type] is None:
            # Try to load from saved .pkl
            if load_if_available:
                logger.info(f"{data_type} not in memory. Attempting to load from .pkl...")
                self.load_from_saved_dataframes([data_type])

            # Try to load from raw .mat
            if (data_type not in self.raw_data or self.raw_data[data_type] is None) and fallback_to_mat:
                logger.warning(f"{data_type} still not loaded. Attempting raw .mat load...")
                logger.info(f"Loading {data_type} from raw .mat files for session={session_name}, run={run_number}, agent={agent}")
                self.load_raw_data_from_mat_files(
                    data_types=[data_type],
                    session_filter=session_name,
                    run_filter=run_number,
                    agent_filter=agent if agent else None
                )

        # Final check: do we have the data now?
        if data_type not in self.raw_data or self.raw_data[data_type] is None:
            logger.warning(f"{data_type} not available in memory.")
            return None

        df = self.raw_data[data_type]
        if session_name is not None:
            df = df[df["session_name"] == session_name]
        if run_number is not None:
            df = df[df["run_number"] == run_number]
        if agent is not None and "agent" in df.columns:
            df = df[df["agent"] == agent]

        return df if not df.empty else None


    def prune_and_interpolate(self):
        """
        Cleans timeline, position, and pupil data:
        - Removes invalid timeline entries (NaNs).
        - Interpolates missing values using sliding or linear interpolation.
        - Overwrites self.raw_data with cleaned DataFrames.
        """
        df_positions = self._get_or_load_data("positions")
        df_pupils = self._get_or_load_data("pupil")
        df_timeline = self._get_or_load_data("neural_timeline")

        cleaned_pos_rows = []
        cleaned_pupil_rows = []
        cleaned_timeline_rows = []

        grouped = df_timeline.groupby(['session_name', 'run_number'])
        logger.info(f"Processing {len(grouped)} session-run groups...")

        for i, ((session, run), time_group) in enumerate(grouped, 1):
            timeline, valid_idx = self._prune_timeline_group(time_group)
            if timeline is None:
                continue

            cleaned_timeline_rows.append(time_group.assign(neural_timeline=[timeline]))

            pos_subset = df_positions.query("session_name == @session and run_number == @run")
            pupil_subset = df_pupils.query("session_name == @session and run_number == @run")

            for agent in ['m1', 'm2']:
                pos_row = self._get_row(pos_subset, agent)
                pupil_row = self._get_row(pupil_subset, agent)
                if pos_row is None or pupil_row is None:
                    continue

                pos_row = self._interpolate_position_row(pos_row, valid_idx, session, run, agent)
                pupil_row = self._interpolate_pupil_row(pupil_row, valid_idx, session, run, agent)

                cleaned_pos_rows.append(pos_row)
                cleaned_pupil_rows.append(pupil_row)

            if i % 50 == 0 or i == len(grouped):
                logger.info(f"Processed {i}/{len(grouped)} groups...")

        self.raw_data["neural_timeline"] = pd.concat(cleaned_timeline_rows, ignore_index=True)
        self.raw_data["positions"] = pd.DataFrame(cleaned_pos_rows)
        self.raw_data["pupil"] = pd.DataFrame(cleaned_pupil_rows)

    # -----------------------------
    # 3. MAT file internals
    # -----------------------------


    def _extract_aligned_struct(self, mat_data: dict):
        """
        Extracts the aligned data structure from the loaded .mat dictionary.

        Args:
            mat_data (dict): Parsed MATLAB contents.

        Returns:
            ndarray or None: Aligned struct if available.
        """
        for key in ['var', 'aligned_position_file']:
            if key in mat_data:
                return mat_data[key][0][0]
        return None

    def _extract_roi_dict(self, agent_roi_data) -> Optional[Dict[str, List[float]]]:
        """
        Extracts bounding boxes for each ROI from a MATLAB struct.

        Args:
            agent_roi_data (np.void): Structured array from .mat file.

        Returns:
            dict: Mapping from ROI name to bounding box coordinates.
        """
        if agent_roi_data is None:
            return None
        return {
            roi_name: agent_roi_data[roi_name][0][0][0]
            for roi_name in agent_roi_data.dtype.names
        }

    # Internal .mat file processing

    def _process_position_file(self, mat_file, session_name: str, run_number: str, agent: str) -> Optional[pd.DataFrame]:
        """
        Parses and formats raw gaze data from a .mat file into a DataFrame.

        Args:
            mat_file: Path to .mat file.
            session_name (str): Session name.
            run_number (str): Run identifier.
            agent (str): 'm1' or 'm2' (if applicable).

        Returns:
            Optional[pd.DataFrame]: Single-row DataFrame or None if unavailable.
        """
        mat_data = load_mat_from_path(mat_file)
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
        """
        Parses and formats raw gaze data from a .mat file into a DataFrame.

        Args:
            mat_file: Path to .mat file.
            session_name (str): Session name.
            run_number (str): Run identifier.
            agent (str): 'm1' or 'm2' (if applicable).

        Returns:
            Optional[pd.DataFrame]: Single-row DataFrame or None if unavailable.
        """
        mat_data = load_mat_from_path(mat_file)
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
        """
        Parses and formats raw gaze data from a .mat file into a DataFrame.

        Args:
            mat_file: Path to .mat file.
            session_name (str): Session name.
            run_number (str): Run identifier.
            agent (str): 'm1' or 'm2' (if applicable).

        Returns:
            Optional[pd.DataFrame]: Single-row DataFrame or None if unavailable.
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

    def _process_roi_rects_file(self, mat_file, session_name: str, run_number: str, agent: str) -> Optional[pd.DataFrame]:
        """
        Parses and formats raw gaze data from a .mat file into a DataFrame.

        Args:
            mat_file: Path to .mat file.
            session_name (str): Session name.
            run_number (str): Run identifier.
            agent (str): 'm1' or 'm2' (if applicable).

        Returns:
            Optional[pd.DataFrame]: Single-row DataFrame or None if unavailable.
        """
        mat_data = load_mat_from_path(mat_file)
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
    # 4. Pruning & interpolation helpers
    # -----------------------------

    def _get_or_load_data(self, data_type: str) -> Optional[pd.DataFrame]:
        """
        Returns the requested data type from memory if available;
        otherwise, attempts to load it using `get_data`.

        This is a convenience wrapper to avoid repetitive checking
        of `self.raw_data.get(...)` followed by a call to `get_data(...)`.

        Args:
            data_type (str): One of the behavioral data types (e.g., 'positions', 'pupil', etc.).

        Returns:
            Optional[pd.DataFrame]: The DataFrame for the requested data type, or None if unavailable.
        """
        df = self.raw_data.get(data_type)
        return df if df is not None else self.get_data(data_type)


    def _prune_timeline_group(self, time_group):
        """
        Removes NaNs from the timeline of a given session/run group.

        Args:
            time_group (pd.DataFrame): Grouped DataFrame with timeline data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Pruned timeline and valid index array.
        """
        idx = time_group.index[0]
        timeline = time_group.loc[idx, 'neural_timeline']
        if timeline is None or len(timeline) == 0:
            return None, None

        timeline = np.array(timeline)
        valid_idx = np.where(~np.isnan(timeline))[0]
        if len(valid_idx) == 0:
            return None, None
        return timeline[valid_idx], valid_idx


    def _get_row(self, df, agent):
        """
        Retrieves the single row for a given agent from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with 'agent' column.
            agent (str): 'm1' or 'm2'.

        Returns:
            Optional[pd.Series]: Matching row or None if not found.
        """
        try:
            return df[df['agent'] == agent].iloc[0]
        except IndexError:
            return None


    def _interpolate_nans(self, array, kind='linear', window_size=10, max_nans=3):
        """
        Interpolates NaN values in a 1D or 2D array.

        Args:
            array (np.ndarray): Input array.
            kind (str): 'linear' or 'sliding'.
            window_size (int): Window size for 2D interpolation.
            max_nans (int): Max NaNs allowed in a window for interpolation.

        Returns:
            np.ndarray: Interpolated array.
        """
        if array.ndim == 1 or kind == 'linear':
            mask = np.isnan(array)
            if np.any(mask) and np.any(~mask):
                array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), array[~mask])
            return array

        elif array.ndim == 2 and kind == 'sliding':
            num_points, num_dims = array.shape
            stride = max_nans
            global_nan_mask = np.isnan(array).any(axis=1)
            for start in range(0, num_points - window_size + 1, stride):
                end = start + window_size
                window_mask = global_nan_mask[start:end]
                nan_count = np.sum(window_mask)
                if 0 < nan_count <= max_nans:
                    window = array[start:end].copy()
                    for col in range(num_dims):
                        col_vals = window[:, col]
                        valid = np.where(~np.isnan(col_vals))[0]
                        if len(valid) > 1:
                            interp_func = interp1d(valid, col_vals[valid], kind='cubic', fill_value="extrapolate", bounds_error=False)
                            to_fill = np.where(window_mask)[0]
                            col_vals[to_fill] = interp_func(to_fill)
                    array[start:end] = window
            return array
        else:
            raise ValueError("Unsupported interpolation type or array shape.")


    def _interpolate_position_row(self, pos_row, valid_idx, session, run, agent):
        """
        Applies 2D interpolation to (x, y) gaze position row.

        Args:
            pos_row (pd.Series): One row of position data.
            valid_idx (np.ndarray): Indices to keep.
            session, run, agent (str): Identifiers (for future debugging).

        Returns:
            pd.Series: Interpolated row.
        """
        x = np.array(pos_row['x'])[valid_idx]
        y = np.array(pos_row['y'])[valid_idx]
        positions = np.stack([x, y], axis=1)
        positions = self._interpolate_nans(positions, kind='sliding', window_size=10, max_nans=3)
        pos_row['x'] = positions[:, 0]
        pos_row['y'] = positions[:, 1]
        return pos_row


    def _interpolate_pupil_row(self, pupil_row, valid_idx, session, run, agent):
        """
        Applies 1D interpolation to a pupil size row.

        Args:
            pupil_row (pd.Series): One row of pupil data.
            valid_idx (np.ndarray): Indices to keep.
            session, run, agent (str): Identifiers.

        Returns:
            pd.Series: Interpolated row.
        """
        pupil = np.array(pupil_row['pupil_size'])[valid_idx]
        pupil = self._interpolate_nans(pupil, kind='linear')
        pupil_row['pupil_size'] = pupil
        return pupil_row




