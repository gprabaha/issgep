# src/socialgaze/features/fixation_detector.py

from __future__ import annotations

import logging
import math
import pdb
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

from socialgaze.config.fixation_config import FixationConfig
from socialgaze.data.gaze_data import GazeData
from socialgaze.utils.fixation_utils import detect_fixations_and_saccades
from socialgaze.utils.hpc_utils import (
    generate_fixation_job_file,
    submit_dsq_array_job,
    track_job_completion,
)
from socialgaze.utils.loading_utils import load_df_from_pkl
from socialgaze.utils.path_utils import (
    get_behav_binary_vector_path,
    get_fixation_job_result_path,
    get_saccade_job_result_path,
)
from socialgaze.utils.saving_utils import save_df_to_pkl

logger = logging.getLogger(__name__)


class FixationDetector:
    def __init__(self, gaze_data: GazeData, config: FixationConfig):
        self.gaze_data = gaze_data
        self.config = config
        self.fixations: Optional[pd.DataFrame] = None
        self.saccades: Optional[pd.DataFrame] = None
        self.binary_vector_paths: dict = {}

    def detect_fixations_through_hpc_jobs(self):
        logger.info("\n ** Generating fixation and saccade detection jobs...")
        logger.info("Loading positions to generate jobs...")
        config = self.config
        pos_df = self.gaze_data.get_data("positions", load_if_available=True)
        if pos_df is None or pos_df.empty:
            logger.warning("No position data found — aborting job creation.")
            return

        tasks = list(pos_df.groupby(["session_name", "run_number", "agent"]).groups.keys())
        if config.test_single_run:
            logger.info("Test mode enabled: running only one randomly selected job.")
            tasks = [random.choice(tasks)]

        generate_fixation_job_file(tasks=tasks, config=config)
        job_id = submit_dsq_array_job(config=config)
        track_job_completion(job_id)
        
        self._load_fixation_and_saccade_results(tasks)


    def detect_fixations_and_saccades_in_single_run(
        self,
        session_name: str,
        run_number: str,
        agent: str
    ) -> None:
        logger.info(f"\n ** Running fixation detection for session={session_name}, run={run_number}, agent={agent}")
        config = self.config
        pos_df = self.gaze_data.get_data("positions", session_name=session_name, run_number=run_number, agent=agent)
        if pos_df is None:
            logger.warning("No position data found for given run.")
            return
        x = np.array(pos_df["x"].values[0])
        y = np.array(pos_df["y"].values[0])
        positions = np.stack([x, y], axis=1)
        non_nan_chunks, chunk_start_indices = _extract_non_nan_chunks(positions)
        args = [(chunk, start) for chunk, start in zip(non_nan_chunks, chunk_start_indices)]

        logger.info("Detecting fixations and saccads for chunks in serial")
        results = [_detect_fix_sacc_in_chunk(arg) for arg in args]
        
        all_fix_start_stops = np.concatenate([r[0] for r in results], axis=0)
        all_sacc_start_stops = np.concatenate([r[1] for r in results], axis=0)
        all_events = np.vstack((all_fix_start_stops, all_sacc_start_stops))
        all_events = all_events[np.argsort(all_events[:, 0])]
        for i in range(len(all_events) - 1):
            assert all_events[i][1] < all_events[i + 1][0], f"Overlap detected between {all_events[i]} and {all_events[i+1]}"
        temp_dir = self.config.temp_dir
        fixation_df = _build_event_df(all_fix_start_stops, session_name, run_number, agent)
        saccade_df = _build_event_df(all_sacc_start_stops, session_name, run_number, agent)
        logger.info("Head of detected fixation df:")
        logger.info(fixation_df.head())
        logger.info("Head of detected saccade df:")
        logger.info(saccade_df.head())
        fix_path = get_fixation_job_result_path(temp_dir, session_name, run_number, agent)
        sacc_path = get_saccade_job_result_path(temp_dir, session_name, run_number, agent)
        save_df_to_pkl(fixation_df, fix_path)
        save_df_to_pkl(saccade_df, sacc_path)


    def save_dataframes(self):
        """
        Saves fixations and saccades DataFrames to their configured paths,
        if they exist and are not empty.
        """
        if self.fixations is not None and not self.fixations.empty:
            save_df_to_pkl(self.fixations, self.config.fixation_df_path)
            logger.info(f"Saved fixation DataFrame to {self.config.fixation_df_path}")
        else:
            logger.warning("Fixation DataFrame is None or empty — skipping save.")

        if self.saccades is not None and not self.saccades.empty:
            save_df_to_pkl(self.saccades, self.config.saccade_df_path)
            logger.info(f"Saved saccade DataFrame to {self.config.saccade_df_path}")
        else:
            logger.warning("Saccade DataFrame is None or empty — skipping save.")


    def load_dataframes(self, behavior_type: Optional[str] = None):
        """
        Loads fixation and/or saccade DataFrames from their configured paths.
        Logs a warning if the file is missing but does not raise an error.
        """
        if behavior_type == "fixations" or behavior_type is None:
            if self.config.fixation_df_path.exists():
                self.fixations = load_df_from_pkl(self.config.fixation_df_path)
            else:
                logger.warning(f"Fixation DataFrame not found at: {self.config.fixation_df_path}")
                self.fixations = None

        if behavior_type == "saccades" or behavior_type is None:
            if self.config.saccade_df_path.exists():
                self.saccades = load_df_from_pkl(self.config.saccade_df_path)
            else:
                logger.warning(f"Saccade DataFrame not found at: {self.config.saccade_df_path}")
                self.saccades = None


    def get_behavior_data(self, behavior_type: str, session_name=None, run_number=None, agent=None) -> pd.DataFrame:
        if behavior_type not in ["fixations", "saccades"]:
            raise ValueError("Behavior type must be either 'fixations' or 'saccades'")
        df = getattr(self, behavior_type)
        if df == None:
            logger.info(f"{behavior_type} not loaded, attempting to load from disk.")
            self.load_dataframes(behavior_type)
            df = getattr(self, behavior_type)
        if session_name:
            df = df[df["session_name"] == session_name]
        if run_number:
            df = df[df["run_number"] == run_number]
        if agent:
            df = df[df["agent"] == agent]
        return df


    def update_fixation_locations(self):
        """
        Annotates each fixation event with the ROI location label, based on average gaze coordinates.
        """
        logger.info("\n ** Updating locations of fixations in self.fixations")
        if self.fixations is None or self.fixations.empty:
            logger.info("Fixation dataframe not loaded yet. Attempting to load from disk.")
            self.load_dataframes("fixations")

        self.fixations["location"] = None

        self.gaze_data.load_from_saved_dataframes(["positions", "roi_vertices"])

        tasks = list(self.fixations.groupby(["session_name", "run_number", "agent"]).groups.items())

        if self.config.use_parallel:
            logger.info(f"Running fixation location update in parallel with {self.config.num_cpus} CPUs")
            with Pool(self.config.num_cpus) as pool:
                func = partial(_annotate_fixation_rows, fixations_df=self.fixations, gaze_data=self.gaze_data)
                results = list(tqdm(pool.imap(func, tasks), total=len(tasks)))
        else:
            results = []
            for task in tqdm(tasks, desc="Annotating fixations"):
                results.append(_annotate_fixation_rows(task, self.fixations, self.gaze_data))

        # Apply results
        self.fixations = _apply_updates_to_df(self.fixations, results, colname="location")


    def update_saccade_from_to(self):
        """
        Annotates each saccade event with the 'from' and 'to' ROI labels based on gaze start/stop points.
        """
        logger.info("\n ** Updating from and to locations of saccades in self.saccades")
        if self.saccades is None or self.saccades.empty:
            logger.info("Saccade dataframe not loaded yet. Attempting to load from disk.")
            self.load_dataframes("saccades")

        self.saccades["from"] = None
        self.saccades["to"] = None

        self.gaze_data.load_from_saved_dataframes(["positions", "roi_vertices"])

        tasks = list(self.saccades.groupby(["session_name", "run_number", "agent"]).groups.items())

        if self.config.use_parallel:
            logger.info(f"Running saccade ROI annotation in parallel with {self.config.num_cpus} CPUs")
            with Pool(self.config.num_cpus) as pool:
                func = partial(_annotate_saccade_rows, saccades_df=self.saccades, gaze_data=self.gaze_data)
                results = list(tqdm(pool.imap(func, tasks), total=len(tasks)))
        else:
            results = []
            for task in tqdm(tasks, desc="Annotating saccades"):
                results.append(_annotate_saccade_rows(task, self.saccades, self.gaze_data))

        self.saccades = _apply_updates_to_df(self.saccades, results, colname=["from", "to"])


    def reconcile_fixation_saccade_label_mismatches(self):
        logger.info("\n ** Reconciling fixation-saccade location mismatches...")
        self._ensure_fix_and_saccade_data_is_loaded_and_labelled()

        fixation_groups = self.fixations.groupby(["session_name", "run_number", "agent"])
        saccade_groups = self.saccades.groupby(["session_name", "run_number", "agent"])
        all_keys = set(fixation_groups.groups.keys()).intersection(saccade_groups.groups.keys())
        log_interval = getattr(self.config, "fixation_labeling_log_interval", 100)

        for idx, key in enumerate(sorted(all_keys)):
            if idx % log_interval == 0:
                logger.info("Processing alignment group %d / %d", idx, len(all_keys))

            changes_made = True
            while changes_made:
                changes_made = self._align_fixation_saccade_pair_for_key(key, fixation_groups, saccade_groups)


    def add_fixation_category_column(self):
        """
        Adds a 'category' column to the fixations DataFrame with values: 'face', 'object', or 'out_of_roi'.
        """
        logger.info("\n ** Categorizing fixations as 'face', 'object', or 'out_of_roi'")
        if self.fixations is None or self.fixations.empty:
            logger.info("Fixation dataframe not loaded yet. Attempting to load from disk.")
            self.load_dataframes("fixations")

        if "location" not in self.fixations.columns:
            logger.info("Fixation locations not found. Running update_fixation_locations().")
            self.update_fixation_locations()

        self.fixations["category"] = self.fixations["location"].apply(_categorize_fixations)

    def add_saccade_category_columns(self):
        """
        Adds simplified 'from_category' and 'to_category' columns to the saccades DataFrame.
        Categories are one of: 'face', 'object', or 'out_of_roi'.
        """
        logger.info("\n ** Categorizing saccade endpoints as 'face', 'object', or 'out_of_roi'")

        if self.saccades is None or self.saccades.empty:
            logger.info("Saccade dataframe not loaded yet. Attempting to load from disk.")
            self.load_dataframes("saccades")

        if "from" not in self.saccades.columns or "to" not in self.saccades.columns:
            self.update_saccade_from_to()

        self.saccades["from_category"] = self.saccades["from"].apply(_categorize_fixations)
        self.saccades["to_category"] = self.saccades["to"].apply(_categorize_fixations)


    def generate_and_save_binary_vectors(
        self,
        behavior_type: Optional[str] = None,
        return_df: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Generates and saves binary vectors for a specific behavior_type, or all if behavior_type is None.
        Stores only the path to each result in self.binary_vector_paths (not the data itself).
        """
        behavior_types = (
            [behavior_type]
            if behavior_type is not None
            else self.config.binary_vector_types_to_generate
        )
        last_df = None
        for btype in behavior_types:
            logger.info(f"\n ** Generating binary vectors for behavior: {btype}")
            result_df = self._generate_binary_vector_df_for_behavior(btype)

            if result_df is None or result_df.empty:
                logger.warning(f"{btype} binary vector dataframe is empty — skipping save.")
                continue
            out_path = get_behav_binary_vector_path(self.config, btype)
            save_df_to_pkl(result_df, out_path)
            self.binary_vector_paths[btype] = out_path
            logger.info("Saved %s binary vectors to %s", btype, out_path)
            last_df = result_df
        if return_df:
            return last_df


    def get_binary_vector_df(self, behavior_type: str) -> pd.DataFrame:
        """
        Loads and returns the binary vector dataframe for the specified behavior type.
        Falls back to constructing the path if it's not already tracked in self.binary_vector_paths.
        """
        path = self.binary_vector_paths.get(behavior_type)
        if path is None:
            path = get_behav_binary_vector_path(self.config, behavior_type)
        if not path.exists():
            raise FileNotFoundError(f"Binary vector file for {behavior_type} not found at: {path}")
        logger.info(f"Loading {behavior_type} binary vector dataframe from {path}")
        return load_df_from_pkl(path)


    # -------------------
    # Helper methods
    # -------------------

    def _load_fixation_and_saccade_results(self, tasks: List[Tuple[str, str, str]]):
        """
        Loads fixation and saccade DataFrames from the temp directory for each (session, run, agent) task.
        After loading, cleans up the temp directory.
        """
        fix_dfs, sacc_dfs = [], []
        temp_dir = self.config.temp_dir
        logger.info("Loading fixation and saccade dfs exported from array jobs")
        for session, run, agent in tasks:
            fix_path = get_fixation_job_result_path(temp_dir, session, run, agent)
            sacc_path = get_saccade_job_result_path(temp_dir, session, run, agent)
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


    def _apply_updates_to_df(df: pd.DataFrame, grouped_results: List[List[tuple]], colname: Union[str, List[str]]) -> pd.DataFrame:
        flat_results = [item for sublist in grouped_results for item in sublist]
        for result in flat_results:
            if isinstance(colname, str):
                idx, value = result
                df.at[idx, colname] = value
            elif isinstance(colname, list):
                idx = result[0]
                for i, col in enumerate(colname):
                    df.at[idx, col] = result[i + 1]
        return df


    def _ensure_fix_and_saccade_data_is_loaded_and_labelled(self):
        if self.fixations is None or self.fixations.empty:
            logger.info("Fixation dataframe not loaded. Attempting to load from disk.")
            self.load_dataframes("fixations")
        if self.saccades is None or self.saccades.empty:
            logger.info("Saccade dataframe not loaded. Attempting to load from disk.")
            self.load_dataframes("saccades")
        if "location" not in self.fixations.columns:
            logger.info("'location' column missing in fixations. Running update_fixation_locations().")
            self.update_fixation_locations()
        if "from" not in self.saccades.columns or "to" not in self.saccades.columns:
            logger.info("'from' or 'to' column missing in saccades. Running update_saccade_from_to().")
            self.update_saccade_from_to()


    def _align_fixation_saccade_pair_for_key(self, key, fixation_groups, saccade_groups) -> bool:
        session_name, run_number, agent = key
        fix_group = fixation_groups.get_group(key).copy()
        sacc_group = saccade_groups.get_group(key).copy()

        # Create indexed event lists
        fix_starts = fix_group["start"].tolist()
        fix_stops = fix_group["stop"].tolist()
        fix_locs = fix_group["location"].tolist()
        fix_indices = fix_group.index.tolist()

        sacc_starts = sacc_group["start"].tolist()
        sacc_stops = sacc_group["stop"].tolist()
        sacc_froms = sacc_group["from"].tolist()
        sacc_tos = sacc_group["to"].tolist()
        sacc_indices = sacc_group.index.tolist()

        events = _merge_and_sort_gaze_events(fix_starts, fix_stops, sacc_starts, sacc_stops, fix_indices, sacc_indices)
        new_fix_locs, new_sacc_froms, new_sacc_tos, changes_made = self._correct_event_label_mismatches(
            session_name, run_number, agent, events, fix_locs, sacc_froms, sacc_tos
        )
        # Write back updates by index
        for idx, val in zip(fix_indices, new_fix_locs):
            self.fixations.at[idx, "location"] = val
        for idx, val_from, val_to in zip(sacc_indices, new_sacc_froms, new_sacc_tos):
            self.saccades.at[idx, "from"] = val_from
            self.saccades.at[idx, "to"] = val_to
        return changes_made


    def _correct_event_label_mismatches(self, session_name, run_number, agent, events,
                                        fixation_locs, saccade_froms, saccade_tos):
        changes_made = False
        for i in range(len(events) - 1):
            start1, end1, type1, index1 = events[i]
            start2, end2, type2, index2 = events[i + 1]
            if start2 - end1 > 100:
                continue
            if type1 == "fixation" and type2 == "saccade":
                fix_lbl = fixation_locs[index1]
                sacc_from_lbl = saccade_froms[index2]
                if set(fix_lbl) != set(sacc_from_lbl):
                    if "out_of_roi" in fix_lbl:
                        logger.info(f"[{session_name}-{run_number}-{agent}] Fixation {index1} out_of_roi → {sacc_from_lbl}")
                        fixation_locs[index1] = sacc_from_lbl
                        changes_made = True
                    elif "out_of_roi" in sacc_from_lbl:
                        logger.info(f"[{session_name}-{run_number}-{agent}] Saccade-from {index2} out_of_roi → {fix_lbl}")
                        saccade_froms[index2] = fix_lbl
                        changes_made = True
            elif type1 == "saccade" and type2 == "fixation":
                sacc_to_lbl = saccade_tos[index1]
                fix_lbl = fixation_locs[index2]
                if set(sacc_to_lbl) != set(fix_lbl):
                    if "out_of_roi" in fix_lbl:
                        logger.info(f"[{session_name}-{run_number}-{agent}] Fixation {index2} out_of_roi → {sacc_to_lbl}")
                        fixation_locs[index2] = sacc_to_lbl
                        changes_made = True
                    elif "out_of_roi" in sacc_to_lbl:
                        logger.info(f"[{session_name}-{run_number}-{agent}] Saccade-to {index1} out_of_roi → {fix_lbl}")
                        saccade_tos[index1] = fix_lbl
                        changes_made = True
        return fixation_locs, saccade_froms, saccade_tos, changes_made


    # == Binary vector generation helpers == #

    def _generate_binary_vector_df_for_behavior(self, behavior_type: str) -> Optional[pd.DataFrame]:
        df, filter_column = self._get_filtered_behavior_dataframe(behavior_type)
        if df is None or df.empty:
            return None

        category_key = behavior_type.replace("_fixation", "").replace("saccade_from_", "").replace("saccade_to_", "")
        df = df[df[filter_column] == category_key]
        if df.empty:
            logger.warning("No events found for %s — skipping", behavior_type)
            return None

        return self._build_binary_vector_dataframe(df, category_key=behavior_type)


    def _get_filtered_behavior_dataframe(self, behavior_type: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        is_fixation = behavior_type.endswith("_fixation")
        is_saccade = behavior_type.startswith("saccade_")

        if is_fixation:
            if self.fixations is None or self.fixations.empty:
                logger.info("Fixation dataframe not loaded yet. Attempting to load from disk.")
                self.load_dataframes("fixations")
            if self.fixations is None or self.fixations.empty:
                logger.warning("No fixation data found — skipping %s", behavior_type)
                return None, None
            if "category" not in self.fixations.columns:
                self.add_fixation_category_column()
            return self.fixations.copy(), "category"

        elif is_saccade:
            if self.saccades is None or self.saccades.empty:
                logger.info("Saccade dataframe not loaded yet. Attempting to load from disk.")
                self.load_dataframes("saccades")
            if self.saccades is None or self.saccades.empty:
                logger.warning("No saccade data found — skipping %s", behavior_type)
                return None, None
            if not {"from", "to", "start", "stop"}.issubset(self.saccades.columns):
                logger.error("Saccade dataframe missing required columns — skipping %s", behavior_type)
                return None, None
            # Ensure categories are present
            if "from_category" not in self.saccades.columns or "to_category" not in self.saccades.columns:
                self.add_saccade_category_columns()
            filter_column = "from_category" if "saccade_from_" in behavior_type else "to_category"
            return self.saccades.copy(), filter_column

        else:
            logger.error("Unrecognized behavior_type: %s", behavior_type)
            return None, None

    def _build_binary_vector_dataframe(self, df: pd.DataFrame, category_key: str) -> pd.DataFrame:
        run_lengths_df = self.gaze_data.get_data("run_lengths")
        vectors = []
        grouped = df.groupby(["session_name", "run_number", "agent"])
        for (session, run, agent), group in grouped:
            run_len_match = run_lengths_df.query("session_name == @session and run_number == @run")
            if run_len_match.empty:
                logger.warning("Run length missing for %s-%s-%s — skipping", session, run, agent)
                continue
            run_length = int(run_len_match["run_length"].values[0])
            binary_vector = np.zeros(run_length, dtype=int)
            for _, row in group.iterrows():
                binary_vector[row["start"]:row["stop"] + 1] = 1
            vectors.append({
                "session_name": session,
                "run_number": run,
                "agent": agent,
                "behavior_type": category_key,
                "binary_vector": binary_vector
            })
        return pd.DataFrame(vectors)


# Fixation and saccade detection functions
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
        "start": events[:, 0],
        "stop": events[:, 1]
    })


# Fixation and saccade labeling functions

def _annotate_fixation_rows(group_task, fixations_df, gaze_data):
    key, indices = group_task
    session, run, agent = key
    group = fixations_df.loc[indices]

    pos_df = gaze_data.get_data("positions", session_name=session, run_number=run, agent=agent)
    roi_df = gaze_data.get_data("roi_vertices", session_name=session, run_number=run, agent=agent)
    if pos_df is None or pos_df.empty or roi_df is None or roi_df.empty:
        return [(idx, None) for idx in group.index]

    x = np.array(pos_df.iloc[0]["x"])
    y = np.array(pos_df.iloc[0]["y"])
    roi_rects = roi_df[["roi_name", "x_min", "y_min", "x_max", "y_max"]]

    updates = []
    for idx, row in group.iterrows():
        start, stop = row["start"], row["stop"]
        mean_x = np.mean(x[start:stop + 1])
        mean_y = np.mean(y[start:stop + 1])
        location = _find_matching_rois((mean_x, mean_y), roi_rects)
        updates.append((idx, location))
    return updates


def _annotate_saccade_rows(group_task, saccades_df, gaze_data):
    key, indices = group_task
    session, run, agent = key
    group = saccades_df.loc[indices]

    pos_df = gaze_data.get_data("positions", session_name=session, run_number=run, agent=agent)
    roi_df = gaze_data.get_data("roi_vertices", session_name=session, run_number=run, agent=agent)
    if pos_df is None or pos_df.empty or roi_df is None or roi_df.empty:
        return [(idx, None, None) for idx in group.index]

    x = np.array(pos_df.iloc[0]["x"])
    y = np.array(pos_df.iloc[0]["y"])
    roi_rects = roi_df[["roi_name", "x_min", "y_min", "x_max", "y_max"]]

    updates = []
    for idx, row in group.iterrows():
        start, stop = row["start"], row["stop"]
        from_pos = (x[start], y[start])
        to_pos = (x[stop], y[stop])
        from_roi = _find_matching_rois(from_pos, roi_rects)
        to_roi = _find_matching_rois(to_pos, roi_rects)
        updates.append((idx, from_roi, to_roi))
    return updates


def _apply_updates_to_df(df: pd.DataFrame, grouped_results: List[List[tuple]], colname: Union[str, List[str]]) -> pd.DataFrame:
    flat_results = [item for sublist in grouped_results for item in sublist]
    for result in flat_results:
        if isinstance(colname, str):
            idx, value = result
            df.at[idx, colname] = value
        elif isinstance(colname, list):
            idx = result[0]
            for i, col in enumerate(colname):
                df.at[idx, col] = result[i + 1]
    return df


def _find_matching_rois(position: np.ndarray, roi_df: pd.DataFrame) -> List[str]:
    matching_rois = []
    for _, roi in roi_df.iterrows():
        if roi["x_min"] <= position[0] <= roi["x_max"] and roi["y_min"] <= position[1] <= roi["y_max"]:
            matching_rois.append(roi["roi_name"])
    return matching_rois if matching_rois else ["out_of_roi"]


def _merge_and_sort_gaze_events(fix_starts, fix_stops, sacc_starts, sacc_stops, fix_indices, sacc_indices):
    events = [(s, e, "fixation", i) for s, e, i in zip(fix_starts, fix_stops, range(len(fix_indices)))]
    events += [(s, e, "saccade", i) for s, e, i in zip(sacc_starts, sacc_stops, range(len(sacc_indices)))]
    events.sort(key=lambda tup: tup[0])
    return events


def _categorize_fixations(location: List[str]) -> str:
    """
    Categorizes a single fixation based on its ROI list.

    Args:
        location (List[str]): List of ROIs the fixation falls into.

    Returns:
        str: One of 'face', 'object', or 'out_of_roi'.
    """
    face_rois = {"face", "mouth", "eyes", "eyes_nf"}
    object_rois = {"left_nonsocial_object", "right_nonsocial_object"}

    roi_set = set(location)
    if roi_set & face_rois:
        return "face"
    elif roi_set & object_rois:
        return "object"
    else:
        return "out_of_roi"




# =========================
# Style / config for plots
# =========================
@dataclass
class FaceFixPlotStyle:
    bin_size_seconds: float = 0.001
    a: float = 1.0
    bar_height: float = 0.28
    colors: Dict[str, str] = None
    per_run_width: float = 4.0
    per_run_height: float = 1.6
    tight_layout: bool = True
    max_cols: int = 5

    # Export options
    export_format: str = "pdf"
    font_family: str = None   # will be auto-selected if not given

    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                "m1": "#1f77b4",             # blue
                "m2": "#2ca02c",             # green
                "m1face_m2face": "#d62728",  # red
                "m1_object": "#9467bd",      # purple
                "m1_out": "#8c564b",         # brown
                "m2_out": "#e377c2",         # pink
                "m1obj_m2face": "#7f7f7f",   # gray
            }

        if self.font_family is None:
            self.font_family = self._choose_font()

    def _choose_font(self) -> str:
        """Pick first available font from a preference list."""
        candidates = ["Arial", "Helvetica", "DejaVu Sans"]
        available = set(fm.get_font_names())
        for cand in candidates:
            if cand in available:
                return cand
        # fallback
        return "sans-serif"



# -------------------------
# NEW: category name keys
# -------------------------
_FACE_KEY = "face_fixation"
_OBJ_KEY  = "object_fixation"          # m1 only (as you said)
_OOR_KEY  = "out_of_roi_fixation"      # assumes this is your key


# -------------------------
# UPDATED method: export every run as PDF (no previews)
# -------------------------
class FixationPlotter(FixationDetector):
    def plot_fixation_timelines(
        self,
        export_pdf_for: Tuple[str, int] | None = None,  # optional: export just one run
        export_dir: Path | None = None,
        style: FaceFixPlotStyle | None = None,
    ) -> None:
        """
        Export per-run fixation timelines as Illustrator-friendly PDFs with 7 rows:
          m1: face, object, out_of_roi
          m2: face, out_of_roi
          combos: (m1_face & m2_face), (m1_object & m2_face)
        """
        style = style or FaceFixPlotStyle()

        # ---- Load binary-vector DFs ----
        face_df = self.get_binary_vector_df(_FACE_KEY)
        obj_df  = self.get_binary_vector_df(_OBJ_KEY)
        try:
            oor_df = self.get_binary_vector_df(_OOR_KEY)
        except Exception:
            oor_df = None  # allow projects that don't have out_of_roi yet

        for df, name in ((face_df, "face_df"), (obj_df, "obj_df")):
            for col in ("session_name", "run_number", "agent", "binary_vector"):
                if col not in df.columns:
                    raise KeyError(f"Required column '{col}' missing in {name}.")
        if oor_df is not None:
            for col in ("session_name", "run_number", "agent", "binary_vector"):
                if col not in oor_df.columns:
                    raise KeyError(f"Required column '{col}' missing in oor_df.")

        # ---- Optionally fetch run lengths ----
        run_lengths_df = None
        try:
            self.gaze_data.load_dataframes(["run_lengths"])
            run_lengths_df = getattr(self.gaze_data, "run_lengths_df", None)
        except Exception:
            pass

        # ---- Collect vectors per (session, run) ----
        recs = _collect_multi_vectors(face_df, obj_df, oor_df)

        # ---- Output root: <plots/fixation_timelines/<today>/<session>> ----
        if export_dir is None:
            export_dir = (self.config.output_dir / "plots" / "fixation_timelines" /
                          datetime.now().strftime("%Y-%m-%d"))

        # ---- Either export one run, or all runs ----
        targets: list[tuple[str, int]]
        if export_pdf_for is not None:
            targets = [(export_pdf_for[0], int(export_pdf_for[1]))]
        else:
            # all keys present in recs
            targets = sorted(recs.keys(), key=lambda x: (x[0], int(x[1])))

        # ---- Export loop ----
        for session, run in tqdm(targets, desc="Plotting run"):
            if (session, run) not in recs:
                logger.warning(f"No vectors at {(session, run)}; skipping.")
                continue

            segs_dict, total_time_sec = _prepare_one_run_multi(
                session=session,
                run=int(run),
                payload=recs[(session, run)],
                run_lengths_df=run_lengths_df,
                bin_size_seconds=style.bin_size_seconds,
            )

            # ensure session subfolder
            session_dir = export_dir / session
            session_dir.mkdir(parents=True, exist_ok=True)

            # base path; extension is added inside exporter
            base = session_dir / f"{session}__run{run}__fixation_timelines"

            _export_seven_band_single_run(
                segs_dict=segs_dict,
                total_time_sec=total_time_sec,
                session=session,
                run=int(run),
                out_basepath=base,
                style=style,
                font_family=style.font_family,
                export_format="pdf",   # force PDF as requested
            )
            logger.info(f"Saved PDF: {base.with_suffix('.pdf')}")

# =========================
# NEW helpers
# =========================
def _collect_multi_vectors(face_df, obj_df, oor_df):
    """
    Build:
      recs[(session, run)] = {
          'm1_face': np.bool_,
          'm2_face': np.bool_,
          'm1_obj' : np.bool_,                # m1 only
          'm1_out' : np.bool_ or None,        # optional if oor_df missing
          'm2_out' : np.bool_ or None,
      }
    """
    recs: dict[tuple[str, int], dict[str, np.ndarray | None]] = {}

    def _assign(df, agent_filter, key_name):
        for _, row in df.iterrows():
            agent = row["agent"]
            if agent not in agent_filter:
                continue
            k = (row["session_name"], int(row["run_number"]))
            recs.setdefault(k, {})
            recs[k][key_name if isinstance(key_name, str) else key_name(agent)] = \
                np.asarray(row["binary_vector"], dtype=bool)

    # faces for both agents
    _assign(face_df, {"m1"}, "m1_face")
    _assign(face_df, {"m2"}, "m2_face")

    # objects: only m1
    _assign(obj_df, {"m1"}, "m1_obj")

    # out-of-roi (if present)
    if oor_df is not None:
        def _key(agent): return "m1_out" if agent == "m1" else "m2_out"
        _assign(oor_df, {"m1", "m2"}, _key)

    # fill missing optional keys with None
    for k in recs:
        for name in ("m1_out", "m2_out"):
            recs[k].setdefault(name, None)

    return recs


def _align_and_truncate(*arrays, L_hint=None):
    """Align by min length (or L_hint if smaller); returns truncated arrays and L."""
    arrays = [a for a in arrays if a is not None]
    if not arrays:
        return [], 0
    L = min(len(a) for a in arrays)
    if L_hint is not None:
        L = min(L, int(L_hint))
    return [a[:L] for a in arrays], L


def _prepare_one_run_multi(
    session: str,
    run: int,
    payload: dict,
    run_lengths_df,
    bin_size_seconds: float,
):
    """
    Build segments for all requested rows and AND-combos.
    Returns:
      segs_dict: dict with keys:
        'm1_face','m1_obj','m1_out','m2_face','m2_out',
        'm1face_m2face','m1obj_m2face'
      total_time_sec
    """
    # Optional run-length cap
    L_cap = None
    if run_lengths_df is not None:
        m = run_lengths_df[
            (run_lengths_df["session_name"] == session) &
            (run_lengths_df["run_number"] == int(run))
        ]
        if not m.empty and "length" in m.columns:
            L_cap = int(m.iloc[0]["length"])

    m1_face = payload.get("m1_face")
    m2_face = payload.get("m2_face")
    m1_obj  = payload.get("m1_obj")
    m1_out  = payload.get("m1_out")
    m2_out  = payload.get("m2_out")

    # Determine master L by existing main tracks (prefer m1_face/m2_face)
    base_vecs = [v for v in (m1_face, m2_face, m1_obj, m1_out, m2_out) if v is not None]
    _, L = _align_and_truncate(*base_vecs, L_hint=L_cap)
    if L == 0:
        raise KeyError(f"No non-empty vectors for {(session, run)}.")

    def _cut(v):
        return None if v is None else v[:L]

    m1_face = _cut(m1_face)
    m2_face = _cut(m2_face)
    m1_obj  = _cut(m1_obj)
    m1_out  = _cut(m1_out)
    m2_out  = _cut(m2_out)

    # Combos
    m1face_m2face = (m1_face & m2_face) if (m1_face is not None and m2_face is not None) else None
    m1obj_m2face  = (m1_obj  & m2_face) if (m1_obj  is not None and m2_face is not None) else None

    # Build segments
    segs_dict = {
        "m1_face": _segments_from_binary(m1_face, bin_size_seconds) if m1_face is not None else [],
        "m1_obj":  _segments_from_binary(m1_obj,  bin_size_seconds) if m1_obj  is not None else [],
        "m1_out":  _segments_from_binary(m1_out,  bin_size_seconds) if m1_out  is not None else [],
        "m2_face": _segments_from_binary(m2_face, bin_size_seconds) if m2_face is not None else [],
        "m2_out":  _segments_from_binary(m2_out,  bin_size_seconds) if m2_out  is not None else [],
        "m1face_m2face": _segments_from_binary(m1face_m2face, bin_size_seconds) if m1face_m2face is not None else [],
        "m1obj_m2face":  _segments_from_binary(m1obj_m2face,  bin_size_seconds) if m1obj_m2face  is not None else [],
    }
    total_time_sec = L * bin_size_seconds
    return segs_dict, total_time_sec


def _segments_from_binary(vec: np.ndarray, bin_size_seconds: float) -> List[Tuple[float, float]]:
    """Convert a 1D binary vector to broken_barh segments: [(start_sec, dur_sec), ...]."""
    v = np.asarray(vec, dtype=np.uint8)
    if v.size == 0:
        return []
    padded = np.pad(v, (1, 1), constant_values=0)
    changes = np.flatnonzero(np.diff(padded))
    starts = changes[::2]
    stops = changes[1::2]  # exclusive
    return [
        (s * bin_size_seconds, (e - s) * bin_size_seconds)
        for s, e in zip(starts, stops) if e > s
    ]


def _export_seven_band_single_run(
    segs_dict: dict,
    total_time_sec: float,
    session: str,
    run: int,
    out_basepath: Path,
    style: FaceFixPlotStyle,
    export_format: str = "pdf",
    font_family: str = "Arial",
) -> Path:
    """
    Illustrator-friendly export: draws each chunk as an individual Rectangle.
    No BrokenBarH, no clipping, no rasterization, transparent background.
    Row order (top->bottom):
      1 m1_face, 2 m1_obj, 3 m1_out, 4 m2_face, 5 m2_out, 6 m1_face&m2_face, 7 m1_obj&m2_face
    """
    # ---- Ensure color keys exist (fallbacks if style not extended) ----
    default_colors = {
        "m1_face":        style.colors.get("m1",            "#3b82f6"),
        "m1_obj":         style.colors.get("m1_object",     "#10b981"),
        "m1_out":         style.colors.get("m1_out",        "#6b7280"),
        "m2_face":        style.colors.get("m2",            "#f59e0b"),
        "m2_out":         style.colors.get("m2_out",        "#a78bfa"),
        "m1face_m2face":  style.colors.get("m1face_m2face", "#ef4444"),
        "m1obj_m2face":   style.colors.get("m1obj_m2face",  "#111827"),
    }

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [font_family],
        "text.usetex": False,

        # Keep text as text (no outlines), avoid masks/clips
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        # Export settings
        "savefig.transparent": True,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.01,
        "path.simplify": False,
    }
    out_path = out_basepath.with_suffix("." + export_format.lower())

    with mpl.rc_context(rc):
        f, ax = plt.subplots(1, 1, figsize=(style.per_run_width, style.per_run_height * 1.25))

        # No background to avoid clipping masks in Illustrator
        ax.patch.set_visible(False)
        f.patch.set_alpha(0.0)
        ax.set_facecolor("none")

        # Geometry
        a, bh = style.a, style.bar_height

        # Top -> bottom rows
        row_order = [
            ("m1_face",       7 * a),
            ("m1_obj",        6 * a),
            ("m1_out",        5 * a),
            ("m2_face",       4 * a),
            ("m2_out",        3 * a),
            ("m1face_m2face", 2 * a),
            ("m1obj_m2face",  1 * a),
        ]

        def _add_rects(segs, y_center, color):
            y0 = y_center - bh / 2.0
            for (x, w) in segs:
                if w <= 0:
                    continue
                rect = mpatches.Rectangle((x, y0), w, bh,
                                          facecolor=color, edgecolor="none", linewidth=0)
                rect.set_clip_on(False)
                rect.set_clip_path(None)
                rect.set_rasterized(False)
                ax.add_patch(rect)

        # Draw rows
        for key, y in row_order:
            segs = segs_dict.get(key, [])
            if segs:
                _add_rects(segs, y, default_colors[key])

        # Axes cosmetics
        ax.set_xlim(0, total_time_sec)
        ax.set_ylim(0, 8 * a)
        ax.set_yticks([y for _, y in row_order])
        ax.set_yticklabels([
            "m1 face", "m1 object", "m1 out-of-ROI",
            "m2 face", "m2 out-of-ROI",
            "m1_face & m2_face",
            "m1_object & m2_face",
        ])
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{session} • run {run}", fontsize=10)
        ax.grid(False)

        # Important: avoid tight_layout to prevent any implicit clip paths
        f.savefig(out_path, format=export_format, dpi=300, transparent=True, metadata={
            "Title": f"{session} run {run} fixation timelines",
            "Subject": "Fixation timelines (m1/m2 face, object, out-of-ROI, and combos).",
        })
        plt.close(f)

    return out_path


