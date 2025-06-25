# src/socialgaze/features/interactivity_detector.py

import os
import logging
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d

from socialgaze.utils.saving_utils import save_df_to_pkl
from socialgaze.utils.loading_utils import load_df_from_pkl
from socialgaze.config.interactivity_config import InteractivityConfig

logger = logging.getLogger(__name__)


class InteractivityDetector:
    def __init__(self, config):
        """
        Initializes the interactivity detector.

        Args:
            config (InteractivityConfig): Configuration object with file paths and parameters.
        """
        self.config = config
        self.fixation_type = config.fixation_type_to_process
        self.output_path = config.mutual_fixation_density_path
        self.num_cpus = config.num_cpus

        self.fix_binary_vector_df: Optional[pd.DataFrame] = None
        self.mutual_fixation_density: Optional[pd.DataFrame] = None
        self.interactivity_periods: Optional[pd.DataFrame] = None


    def detect_mutual_face_fix_density(self, overwrite: bool = False) -> None:
        """
        Computes or loads mutual fixation density and stores it in self.mutual_fixation_density.

        Args:
            overwrite (bool): If True, forces recomputation even if file exists.
        """
        if not overwrite and os.path.exists(self.output_path):
            logger.info(f"Mutual fixation density already exists. Loading from: {self.output_path}")
            self.mutual_fixation_density = self.load_fix_densities()
            return

        if self.fix_binary_vector_df is None:
            logger.info(f"Loading fixation binary vector DataFrame from: {self.config.fix_binary_vector_df_path}")
            self.fix_binary_vector_df = load_df_from_pkl(self.config.fix_binary_vector_df_path)

        logger.info(f"Computing mutual {self.fixation_type} fixation density...")

        session_groups = self.fix_binary_vector_df.groupby('session_name')

        if self.config.use_parallel:
            logger.info(f"Running in parallel using {self.num_cpus} CPUs")
            results = Parallel(n_jobs=self.num_cpus)(
                delayed(self._get_fix_density_in_session)(session_name, session_df)
                for session_name, session_df in tqdm(session_groups, desc="Processing Sessions")
            )
        else:
            logger.info("Running in serial mode")
            results = [
                self._get_fix_density_in_session(session_name, session_df)
                for session_name, session_df in tqdm(session_groups, desc="Processing Sessions")
            ]

        flattened_results = [entry for session_result in results for entry in session_result]
        self.mutual_fixation_density = pd.DataFrame(flattened_results)
        logger.info("Saving mutual fixation density...")
        self.save_fix_densities()



    def get_density(self, session_name: str = None, run_number: str = None) -> pd.DataFrame:
        """
        Returns the computed mutual fixation density, optionally filtered by session or run.
        If not already loaded, attempts to load from the saved pickle file.
        Args:
            session_name (str, optional): Filter by session name.
            run_number (str, optional): Filter by run number.
        Returns:
            pd.DataFrame: Filtered or full mutual fixation density.
        Raises:
            ValueError: If density is neither loaded nor found on disk.
        """
        if self.mutual_fixation_density is None:
            if os.path.exists(self.output_path):
                logger.info(f"Loading mutual fixation density from disk: {self.output_path}")
                self.mutual_fixation_density = self.load_result()
            else:
                raise ValueError(
                    f"Mutual fixation density not available in memory or on disk. "
                    f"Call `.run()` to compute it."
                )
        df = self.mutual_fixation_density
        if session_name:
            df = df[df['session_name'] == session_name]
        if run_number:
            df = df[df['run_number'] == run_number]
        return df.reset_index(drop=True)


    def save_fix_densities(self) -> None:
        """
        Saves the mutual fixation density dataframe to the configured output path.
        The dataframe must be present in `self.mutual_fixation_density`. Raises a warning if it is None or empty.
        """
        if self.mutual_fixation_density is None or self.mutual_fixation_density.empty:
            logger.warning("Mutual fixation density is empty or None. Nothing to save.")
            return
        save_df_to_pkl(self.mutual_fixation_density, self.output_path)
        logger.info(f"Mutual fixation density saved to {self.output_path}")


    def load_fix_densities(self) -> pd.DataFrame:
        """
        Loads the mutual fixation density dataframe from the configured output path.
        Returns:
            pd.DataFrame: The loaded mutual fixation density dataframe.
        Raises:
            FileNotFoundError: If the output file does not exist.
        """
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"No fixation density file found at: {self.output_path}")
        
        self.mutual_fixation_density = load_df_from_pkl(self.output_path)
        logger.info(f"Mutual fixation density loaded from {self.output_path}")
        return self.mutual_fixation_density


    def compute_interactivity_periods(self, overwrite: bool = False) -> None:
        """
        Computes or loads periods of interactivity (start/stop indices) for each session and run,
        using thresholded mutual fixation density. Stores the result in self.interactivity_df.
        Args:
            overwrite (bool): If True, recompute and overwrite existing file. Otherwise, try to load.
        """
        output_path = self.config.interactivity_df_path

        if not overwrite and os.path.exists(output_path):
            logger.info(f"Interactivity periods already exist. Loading from: {output_path}")
            self.interactivity_periods = load_df_from_pkl(output_path)
            return
        if self.mutual_fixation_density is None:
            logger.info("No mutual fixation density in memory, attempting to load.")
            self.mutual_fixation_density = self.load_fix_densities()
        results = []
        for (session, run), group in self.mutual_fixation_density.groupby(["session_name", "run_number"]):
            mutual = np.array(group["mutual_density"].values[0])
            threshold = self.config.interactivity_threshold * np.mean(mutual)
            is_interactive = mutual > threshold
            periods = _get_interactive_periods(is_interactive)
            for start, stop in periods:
                results.append({
                    "session_name": session,
                    "run_number": run,
                    "start": start,
                    "stop": stop
                })
        self.interactivity_periods = pd.DataFrame(results)
        return self.interactivity_periods 


    def save_interactivity_periods(self, path: str = None):
        """Saves the interactivity_df to disk as a pickle file."""
        if self.interactivity_periods is None:
            raise ValueError("No interactivity data to save.")
        path = path or self.config.interactivity_df_path
        save_df_to_pkl(self.interactivity_periods, path)
        logger.info("Saved interactivity dataframe to %s", path)


    def load_interactivity_periods(self, path: str = None) -> pd.DataFrame:
        """Loads the interactivity_df from disk."""
        path = path or self.config.interactivity_df_path
        self.interactivity_periods = load_df_from_pkl(path)
        logger.info("Loaded interactivity dataframe from %s", path)
        return self.interactivity_periods

    def get_interactivity_periods(self, session_name: str = None, run_number: str = None) -> pd.DataFrame:
        """
        Returns the interactivity periods, optionally filtered by session and run.
        Loads from disk if not already in memory.
        """
        if self.interactivity_periods is None:
            logger.info("Interactivity dataframe not in memory. Loading from disk.")
            self.load_interactivity_periods()
        df = self.interactivity_periods
        if session_name:
            df = df[df["session_name"] == session_name]
        if run_number:
            df = df[df["run_number"] == run_number]
        return df.reset_index(drop=True)


    def _get_fix_density_in_session(self, session_name, session_df):
        """
        Computes mutual fixation density for each run in a session.

        For each run, this function retrieves binary fixation vectors for both agents (m1 and m2),
        calculates their fixation durations and inter-fixation intervals (IFIs), and smooths the
        vectors using Gaussian filters whose standard deviation is based on the mean of fixation 
        duration and IFI. The mutual fixation density is then computed as the geometric mean of
        the normalized densities from both agents.

        Args:
            session_name (str): Name of the session being processed.
            session_df (pd.DataFrame): Dataframe containing fixation binary vectors for the session.

        Returns:
            List[Dict]: A list of dictionaries, one per run, containing session metadata,
                        fixation metrics, and density arrays for m1, m2, and mutual fixation.
        """

        run_groups = session_df.groupby("run_number")
        results = []
        for run_number, run_df in run_groups:
            m1 = run_df[(run_df.agent == "m1") & (run_df.fixation_type == self.fixation_type)]
            m2 = run_df[(run_df.agent == "m2") & (run_df.fixation_type == self.fixation_type)]
            if m1.empty or m2.empty:
                continue

            m1_vec = np.array(m1.binary_vector.values[0])
            m2_vec = np.array(m2.binary_vector.values[0])

            m1_fix_dur, m1_ifi = compute_fixation_metrics(m1_vec)
            m2_fix_dur, m2_ifi = compute_fixation_metrics(m2_vec)
            m1_sigma = (m1_fix_dur + m1_ifi) / 2
            m2_sigma = (m2_fix_dur + m2_ifi) / 2

            min_len = min(len(m1_vec), len(m2_vec))
            m1_vec = m1_vec[:min_len]
            m2_vec = m2_vec[:min_len]

            m1_density = gaussian_filter1d(m1_vec.astype(float), sigma=m1_sigma, mode='constant')
            m2_density = gaussian_filter1d(m2_vec.astype(float), sigma=m2_sigma, mode='constant')

            m1_density_norm = normalize_density(m1_density)
            m2_density_norm = normalize_density(m2_density)
            mutual = np.sqrt(m1_density_norm * m2_density_norm)
            mutual_norm = normalize_density(mutual)

            results.append({
                'session_name': session_name,
                'run_number': run_number,
                'fixation_type': self.fixation_type,
                'm1_fix_dur': m1_fix_dur,
                'm2_fix_dur': m2_fix_dur,
                'm1_ifi': m1_ifi,
                'm2_ifi': m2_ifi,
                'm1_sigma': m1_sigma,
                'm2_sigma': m2_sigma,
                'm1_density': list(m1_density_norm),
                'm2_density': list(m2_density_norm),
                'mutual_density': list(mutual_norm)
            })

        return results


def compute_fixation_metrics(binary_vector):
    """
    Computes average fixation duration and inter-fixation interval (IFI) from a binary vector.

    Args:
        binary_vector (np.ndarray or List[int]): A binary vector where 1 indicates a fixation
                                                 and 0 indicates absence of fixation.

    Returns:
        Tuple[float, float]: Mean fixation duration and mean inter-fixation interval.
                             Returns (0, 0) if no fixation is detected.
    """
    vec = np.array(binary_vector)
    if np.all(vec == 0):
        return 0, 0
    if np.all(vec == 1):
        return len(vec), 0
    changes = np.where(np.diff(np.pad(vec, (1, 1), 'constant')) != 0)[0]
    durations = np.diff(changes)
    fix_durations = durations[::2]
    ifi_durations = durations[1::2] if len(durations) > 1 else [0]
    return np.mean(fix_durations), np.mean(ifi_durations)


def normalize_density(arr):
    """
    Normalizes an array to the range [0, 1].
    Args:
        arr (np.ndarray): Input array.
    Returns:
        np.ndarray: Normalized array where the minimum becomes 0 and maximum becomes 1.
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)


def _get_interactive_periods(binary_vector):
    """
    Returns a list of (start, stop) indices where stop is inclusive.
    Safe for use with: v[start:stop+1]
    """
    vec = np.asarray(binary_vector, dtype=int)
    if np.all(vec == 0):
        return []

    padded = np.pad(vec, (1, 1), constant_values=0)
    change_indices = np.flatnonzero(np.diff(padded))

    starts = change_indices[::2]
    stops = change_indices[1::2] - 1  # make 'stop' inclusive

    return list(zip(starts, stops))


