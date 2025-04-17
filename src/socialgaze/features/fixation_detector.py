# src/socialgaze/features/fixation_detector.py

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import random
from multiprocessing import Pool
from pathlib import Path
import shutil

from socialgaze.config.fixation_config import FixationConfig
from socialgaze.data.gaze_data import GazeData
from socialgaze.utils.fixation_utils import detect_fixations_and_saccades
from socialgaze.utils.hpc_utils import (
    generate_fixation_job_file,
    submit_dsq_array_job,
    track_job_completion
)
from socialgaze.utils.loading_utils import load_df_from_pkl
from socialgaze.utils.saving_utils import save_df_to_pkl

logger = logging.getLogger(__name__)

class FixationDetector:
    def __init__(self, gaze_data: GazeData, config: FixationConfig):
        self.gaze_data = gaze_data
        self.config = config
        self.fixations = pd.DataFrame()
        self.saccades = pd.DataFrame()


    def detect_fixations_through_hpc_jobs(self, fixation_config: FixationConfig):
        logger.info("Loading positions to generate fixation detection jobs...")
        pos_df = self.gaze_data.get_data("positions", load_if_available=True)
        if pos_df is None or pos_df.empty:
            logger.warning("No position data found — aborting job creation.")
            return

        tasks = list(pos_df.groupby(["session_name", "run_number", "agent"]).groups.keys())
        if fixation_config.test_single_run:
            logger.info("Test mode enabled: running only one randomly selected job.")
            tasks = [random.choice(tasks)]

        # Setup job output directory
        output_dir = Path(self.config.output_dir)
        jobs_dir = output_dir / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)

        job_file_path = jobs_dir / fixation_config.job_file_name
        script_path = fixation_config.job_script_path

        generate_fixation_job_file(
            tasks=tasks,
            job_file_path=job_file_path,
            script_path=script_path,
            is_grace=self.config.is_grace
        )

        job_id = submit_dsq_array_job(
            job_file_path=job_file_path,
            job_out_dir=jobs_dir,
            job_name=fixation_config.job_name,
            partition=fixation_config.partition,
            cpus=fixation_config.cpus_per_task,
            mem_per_cpu=fixation_config.mem_per_cpu,
            time_limit=fixation_config.time_limit
        )

        track_job_completion(job_id)
        self._load_fixation_and_saccade_results(tasks)


    def _load_fixation_and_saccade_results(self, tasks: List[Tuple[str, str, str]]):
        temp_dir = Path(self.config.processed_data_dir) / "temp"
        fix_dfs, sacc_dfs = [], []

        for session, run, agent in tasks:
            fix_path = temp_dir / f"fixations_{session}_{run}_{agent}.pkl"
            sacc_path = temp_dir / f"saccades_{session}_{run}_{agent}.pkl"
            if fix_path.exists():
                fix_dfs.append(load_df_from_pkl(fix_path))
            else:
                logger.warning("Missing fixation file: %s", fix_path)
            if sacc_path.exists():
                sacc_dfs.append(load_df_from_pkl(sacc_path))
            else:
                logger.warning("Missing saccade file: %s", sacc_path)
        self.fixations = pd.concat(fix_dfs, ignore_index=True) if fix_dfs else pd.DataFrame()
        self.saccades = pd.concat(sacc_dfs, ignore_index=True) if sacc_dfs else pd.DataFrame()
        # Clean up temp folder
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Deleted temporary folder: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to delete temp directory {temp_dir}: {e}")


    def detect_fixations_and_saccades_in_single_run(
        self,
        session_name: str,
        run_number: str,
        agent: str,
        config: FixationConfig
    ) -> None:
        logger.info(f"Running fixation detection for session={session_name}, run={run_number}, agent={agent}")

        pos_df = self.gaze_data.get_data("positions", session_name=session_name, run_number=run_number, agent=agent)
        if pos_df is None:
            logger.warning("No position data found for given run.")
            return
        x = np.array(pos_df["x"].values[0])
        y = np.array(pos_df["y"].values[0])
        positions = np.stack([x, y], axis=1)
        non_nan_chunks, chunk_start_indices = _extract_non_nan_chunks(positions)
        args = [(chunk, start) for chunk, start in zip(non_nan_chunks, chunk_start_indices)]
        if config.use_parallel:
            logger.info("Detecting fixations and saccads for chunks in parallel")
            num_cpus = getattr(config, "num_cpus", 1)
            with Pool(processes=min(16, num_cpus)) as pool:
                results = pool.map(_detect_fix_sacc_in_chunk, args)
        else:
            logger.info("Detecting fixations and saccads for chunks in serial")
            results = [_detect_fix_sacc_in_chunk(arg) for arg in args]
        all_fix_start_stops = np.concatenate([r[0] for r in results], axis=0)
        all_sacc_start_stops = np.concatenate([r[1] for r in results], axis=0)
        all_events = np.vstack((all_fix_start_stops, all_sacc_start_stops))
        all_events = all_events[np.argsort(all_events[:, 0])]
        for i in range(len(all_events) - 1):
            assert all_events[i][1] < all_events[i + 1][0], f"Overlap detected between {all_events[i]} and {all_events[i+1]}"
        temp_dir = Path(self.config.processed_data_dir) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        fixation_df = _build_event_df(all_fix_start_stops, session_name, run_number, agent)
        saccade_df = _build_event_df(all_sacc_start_stops, session_name, run_number, agent)
        save_df_to_pkl(fixation_df, temp_dir / f"fixations_{session_name}_{run_number}_{agent}.pkl")
        save_df_to_pkl(saccade_df, temp_dir / f"saccades_{session_name}_{run_number}_{agent}.pkl")


    def save_dataframes(self):
        save_df_to_pkl(self.fixations, Path(self.config.processed_data_dir) / "fixations.pkl")
        save_df_to_pkl(self.saccades, Path(self.config.processed_data_dir) / "saccades.pkl")


    def load_dataframes(self, behavior_type: Optional[str] = None):
        fix_path = Path(self.config.processed_data_dir) / "fixations.pkl"
        sacc_path = Path(self.config.processed_data_dir) / "saccades.pkl"
        if behavior_type == "fixations" or behavior_type is None:
            if fix_path.exists():
                self.fixations = load_df_from_pkl(fix_path)
            else:
                raise FileNotFoundError(f"Missing {fix_path}")
        if behavior_type == "saccades" or behavior_type is None:
            if sacc_path.exists():
                self.saccades = load_df_from_pkl(sacc_path)
            else:
                raise FileNotFoundError(f"Missing {sacc_path}")


    def get_behavior_data(self, behavior_type: str, session_name=None, run_number=None, agent=None) -> pd.DataFrame:
        if behavior_type not in ["fixations", "saccades"]:
            raise ValueError("Behavior type must be either 'fixations' or 'saccades'")
        df = getattr(self, behavior_type)
        if df.empty:
            logger.info(f"{behavior_type} not loaded, attempting to load from disk.")
            self.load_dataframes(behavior_type)
            df = getattr(self, behavior_type)
        if session_name:
            df = df[df["session_name"] == session_name]
        if run_number:
            df = df[df["run_number"] == run_number]
        if agent:
            df = df[df["agent"] == agent]
        return df[df["agent"] == agent]
        return df


def _extract_non_nan_chunks(positions: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
    non_nan_chunks = []
    start_indices = []
    n = positions.shape[0]
    valid_mask = ~np.isnan(positions).any(axis=1)
    diff = np.diff(valid_mask.astype(int))
    chunk_starts = np.where(diff == 1)[0] + 1
    chunk_ends = np.where(diff == -1)[0] + 1
    if valid_mask[0]:
        chunk_starts = np.insert(chunk_starts, 0, 0)
    if valid_mask[-1]:
        chunk_ends = np.append(chunk_ends, n)
    for start, end in zip(chunk_starts, chunk_ends):
        non_nan_chunks.append(positions[start:end])
        start_indices.append(start)
    return non_nan_chunks, start_indices

def _detect_fix_sacc_in_chunk(args: Tuple[np.ndarray, int]) -> Tuple[np.ndarray, np.ndarray]:
    position_chunk, start_ind = args
    fixation_start_stop_indices, saccade_start_stop_indices = detect_fixations_and_saccades(position_chunk)
    fixation_start_stop_indices += start_ind
    saccade_start_stop_indices += start_ind
    return fixation_start_stop_indices, saccade_start_stop_indices

def _build_event_df(events: np.ndarray, session_name: str, run_number: str, agent: str) -> pd.DataFrame:
    return pd.DataFrame({
        "session_name": session_name,
        "run_number": run_number,
        "agent": agent,
        "starts": events[:, 0],
        "stops": events[:, 1]
    })
