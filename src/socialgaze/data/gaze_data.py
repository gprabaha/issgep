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

import pdb

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
        self.positions: Optional[pd.DataFrame] = None
        self.pupil: Optional[pd.DataFrame] = None
        self.roi_vertices: Optional[pd.DataFrame] = None
        self.neural_timeline: Optional[pd.DataFrame] = None
        self.run_lengths: Optional[pd.DataFrame] = None


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
            loaded_data = []
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
                                loaded_data.append(df)
                    else:
                        path = path_func(self.config, session_name, run_number)
                        df = process_func(path, session_name, run_number)
                        if df is not None:
                            loaded_data.append(df)
            if loaded_data:
                setattr(self, data_type, pd.concat(loaded_data, ignore_index=True))

    # Load and save generated dataframes
    
    def load_dataframes(self, data_types: Optional[List[str]] = None):
        """
        Loads previously saved DataFrames from .pkl files into self.raw_data.

        Args:
            data_types (List[str], optional): Data types to load. If None, all are loaded.
        """
        df_path_map = {
            'positions': self.config.positions_df_path,
            'pupil': self.config.pupil_df_path,
            'roi_vertices': self.config.roi_vertices_df_path,
            'neural_timeline': self.config.neural_timeline_df_path
        }
        data_types = data_types if data_types else self.behav_data_loader_dict.keys()
        for data_type in data_types:
            pkl_path = df_path_map.get(data_type)
            if pkl_path.exists():
                try:
                    df = load_df_from_pkl(pkl_path)
                    setattr(self, data_type, df)
                    logger.info(f"Loaded {data_type} from {pkl_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {data_type} from disk: {e}")
        # Also try loading run_lengths if present
        run_lengths_path = self.config.run_length_df_path
        if run_lengths_path.exists():
            try:
                self.run_lengths = load_df_from_pkl(run_lengths_path)
                logger.info("Loaded run_lengths from disk.")
            except Exception as e:
                logger.warning(f"Failed to load run_lengths.pkl: {e}")



    def save_as_dataframes(self, output_dir: Optional[Path] = None):
        """
        Saves all in-memory DataFrames in self.raw_data and run_lengths to the specified output directory as .pkl files.

        Args:
            output_dir (Optional[Path]): Directory to save the pickle files in. Defaults to self.config.processed_data_dir.
        """
        output_dir = output_dir or self.config.processed_data_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving raw data dataframes...")
        data_path_map = {
            'positions': (self.positions, self.config.positions_df_path),
            'pupil': (self.pupil, self.config.pupil_df_path),
            'roi_vertices': (self.roi_vertices, self.config.roi_vertices_df_path),
            'neural_timeline': (self.neural_timeline, self.config.neural_timeline_df_path)
        }
        for data_type, (df, path) in data_path_map.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                save_df_to_pkl(df, path)

        if self.run_lengths is not None and isinstance(self.run_lengths, pd.DataFrame):
            run_lengths_path = self.config.run_length_df_path
            save_df_to_pkl(self.run_lengths, run_lengths_path)
            logger.info("Saved run_lengths to disk.")
        logger.info("All raw data (and run lengths) saved as DataFrames.")


    # Fetch any part of the data
    def get_data(self,
                data_type: str,
                session_name: Optional[str] = None,
                run_number: Optional[str] = None,
                agent: Optional[str] = None,
                load_if_available: bool = True,
                fallback_to_mat: bool = False) -> Optional[pd.DataFrame]:
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
        df = getattr(self, data_type, None)
        # Try loading from .pkl if missing
        if df is None and load_if_available:
            logger.info(f"{data_type} not in memory. Attempting to load from .pkl...")
            self.load_dataframes([data_type])
            df = getattr(self, data_type, None)

        # Try loading from .mat if still missing
        if df is None and fallback_to_mat:
            logger.warning(f"{data_type} still not loaded. Attempting raw .mat load...")
            logger.info(f"Loading {data_type} from raw .mat files for session={session_name}, run={run_number}, agent={agent}")
            self.load_raw_data_from_mat_files(
                data_types=[data_type],
                session_filter=session_name,
                run_filter=run_number,
                agent_filter=agent if agent else None
            )
            df = getattr(self, data_type, None)
        
        if df is None:
            logger.warning(f"{data_type} not available in memory.")
            return None
        # Apply filters
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
        - Updates self.positions, self.pupil, and self.neural_timeline with cleaned DataFrames.
        """
        df_positions = self.get_data("positions")
        df_pupils = self.get_data("pupil")
        df_timeline = self.get_data("neural_timeline")

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

        self.neural_timeline = pd.concat(cleaned_timeline_rows, ignore_index=True)
        self.positions = pd.DataFrame(cleaned_pos_rows)
        self.pupil = pd.DataFrame(cleaned_pupil_rows)



    def get_run_lengths(self) -> pd.DataFrame:
        """
        Returns the run_lengths DataFrame.
        Loads from disk or computes from neural timeline if necessary.
        Raises error if neither is available.
        """
        if self.run_lengths is not None:
            return self.run_lengths
        # Attempt to load from disk
        path = self.config.run_length_df_path
        if path.exists():
            try:
                self.run_lengths = load_df_from_pkl(path)
                logger.info("Loaded run_lengths from disk.")
                return self.run_lengths
            except Exception as e:
                logger.warning(f"Failed to load run_lengths from disk: {e}")
        # Attempt to compute from raw_data
        if self.neural_timeline is None:
            self.run_lengths = self._compute_run_lengths_from_timeline()
            return self.run_lengths
        raise RuntimeError("Cannot retrieve run lengths — no timeline data in memory and no saved .pkl found.")


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


    def _compute_run_lengths_from_timeline(self) -> pd.DataFrame:
        """
        Computes the number of timepoints in each (session_name, run_number) based on the neural timeline.

        Returns:
            pd.DataFrame: DataFrame with columns ['session_name', 'run_number', 'run_length']
        """
        if self.neural_timeline is None:
            raise ValueError("Neural timeline data not loaded. Cannot compute run lengths.")
        
        df = self.neural_timeline
        rows = []
        for (session, run), group in df.groupby(['session_name', 'run_number']):
            try:
                timeline = group.iloc[0]['neural_timeline']
                length = len(timeline) if timeline is not None else 0
                rows.append({
                    'session_name': session,
                    'run_number': run,
                    'run_length': length
                })
            except Exception as e:
                logger.warning(f"Failed to extract run length for {session}, run {run}: {e}")
        
        return pd.DataFrame(rows)

