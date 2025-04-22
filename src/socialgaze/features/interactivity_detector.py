# src/socialgaze/features/interactivity_detector.py

import os
import logging
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
    def __init__(self, fix_binary_vector_df: pd.DataFrame, config: InteractivityConfig):
        self.fix_binary_vector_df = fix_binary_vector_df
        self.config = config
        self.fixation_type = config.fixation_type_to_process
        self.output_path = config.mutual_fixation_density_path
        self.num_cpus = config.num_cpus
        self.mutual_fixation_density: pd.DataFrame = None  # Will be populated after run()

    def run(self, overwrite: bool = False) -> None:
        """
        Computes or loads mutual fixation density and stores it in self.mutual_fixation_density.
        """
        if not overwrite and os.path.exists(self.output_path):
            logger.info(f"Mutual fixation density already exists. Loading from: {self.output_path}")
            self.mutual_fixation_density = self.load_result()
            return

        logger.info(f"Computing mutual {self.fixation_type} fixation density...")

        session_groups = self.fix_binary_vector_df.groupby('session_name')

        if self.config.use_parallel:
            logger.info(f"Running in parallel using {self.num_cpus} CPUs")
            results = Parallel(n_jobs=self.num_cpus)(
                delayed(self._process_session)(session_name, session_df)
                for session_name, session_df in tqdm(session_groups, desc="Processing Sessions")
            )
        else:
            logger.info("Running in serial mode")
            results = [
                self._process_session(session_name, session_df)
                for session_name, session_df in tqdm(session_groups, desc="Processing Sessions")
            ]
        logger.info("Saving mutual density detection results...")
        flattened_results = [entry for session_result in results for entry in session_result]
        self.mutual_fixation_density = pd.DataFrame(flattened_results)
        self.save_result(self.mutual_fixation_density)



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


    def save_result(self, df: pd.DataFrame):
        save_df_to_pkl(df, self.output_path)

    def load_result(self) -> pd.DataFrame:
        return load_df_from_pkl(self.output_path)

    def _process_session(self, session_name, session_df):
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
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
