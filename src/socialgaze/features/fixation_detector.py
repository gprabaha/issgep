# src/socialgaze/features/fixation_detector.py

import pdb

import logging
from typing import List, Tuple, Optional, Union
from collections import defaultdict
from pathlib import Path
import random
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import shutil
import numpy as np
import pandas as pd

from socialgaze.config.fixation_config import FixationConfig
from socialgaze.data.gaze_data import GazeData
from socialgaze.utils.fixation_utils import detect_fixations_and_saccades
from socialgaze.utils.hpc_utils import (
    generate_fixation_job_file,
    submit_dsq_array_job,
    track_job_completion
)
from socialgaze.utils.path_utils import get_fixation_job_result_path, get_saccade_job_result_path
from socialgaze.utils.loading_utils import load_df_from_pkl
from socialgaze.utils.saving_utils import save_df_to_pkl

logger = logging.getLogger(__name__)

class FixationDetector:
    def __init__(self, gaze_data: GazeData, config: FixationConfig):
        self.gaze_data = gaze_data
        self.config = config
        self.fixations: Optional[pd.DataFrame] = None
        self.saccades: Optional[pd.DataFrame] = None
        self.fixation_binary_vectors: Optional[pd.DataFrame] = None

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
        Raises error if the file is missing.
        """
        if behavior_type == "fixations" or behavior_type is None:
            if self.config.fixation_df_path.exists():
                self.fixations = load_df_from_pkl(self.config.fixation_df_path)
            else:
                raise FileNotFoundError(f"Missing {self.config.fixation_df_path}")

        if behavior_type == "saccades" or behavior_type is None:
            if self.config.saccade_df_path.exists():
                self.saccades = load_df_from_pkl(self.config.saccade_df_path)
            else:
                raise FileNotFoundError(f"Missing {self.config.saccade_df_path}")


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


    def generate_fixation_binary_vectors(self, return_df: bool = False) -> Optional[pd.DataFrame]:
        """
        Generates binary vectors for each fixation category per run and agent.
        Stores the result in self.fixation_binary_vectors and optionally returns it.

        Args:
            return_df (bool): If True, returns the binary vector DataFrame.
        Returns:
            Optional[pd.DataFrame]: Long-format DataFrame of binary vectors (if return_df=True).
        """
        logger.info("\n ** Generating fixation binary vectors...")

        if self.fixations is None or self.fixations.empty:
            logger.info("Fixation dataframe not loaded yet. Attempting to load from disk.")
            self.load_dataframes("fixations")

        if "category" not in self.fixations.columns:
            logger.info("'category' column not found in fixations. Running add_fixation_category_column().")
            self.add_fixation_category_column()

        run_lengths_df = self.gaze_data.get_data('run_lengths')
        vectors = []

        grouped = self.fixations.groupby(["session_name", "run_number", "agent"])
        for (session, run, agent), group in grouped:
            run_len_match = run_lengths_df.query("session_name == @session and run_number == @run")
            if run_len_match.empty:
                logger.warning("Run length missing for %s-%s-%s — skipping", session, run, agent)
                continue
            run_length = int(run_len_match["run_length"].values[0])

            binary_dict = defaultdict(lambda: np.zeros(run_length, dtype=int))

            for _, row in group.iterrows():
                start, stop = row["start"], row["stop"]
                category = row["category"]
                binary_dict[category][start:stop + 1] = 1

            for cat, vec in binary_dict.items():
                vectors.append({
                    "session_name": session,
                    "run_number": run,
                    "agent": agent,
                    "fixation_type": cat,
                    "binary_vector": vec
                })

        result_df = pd.DataFrame(vectors)
        self.fixation_binary_vectors = result_df
        logger.info("Fixation binary vector generation complete.")
        if return_df:
            return result_df


    def save_fixation_binary_vectors(self):
        """
        Saves the fixation_binary_vectors DataFrame to disk if it exists and is not empty.
        """
        if self.fixation_binary_vectors is not None and not self.fixation_binary_vectors.empty:
            path = self.config.fix_binary_vec_df_path
            save_df_to_pkl(self.fixation_binary_vectors, path)
            logger.info(f"Saved fixation binary vectors to {path}")
        else:
            logger.warning("Fixation binary vectors not available or empty — skipping save.")


    def load_fixation_binary_vectors(self):
        """
        Loads the fixation_binary_vectors DataFrame from disk if available.
        """
        path = self.config.fix_binary_vec_df_path
        if path.exists():
            self.fixation_binary_vectors = load_df_from_pkl(path)
            logger.info(f"Loaded fixation binary vectors from {path}")
        else:
            logger.warning(f"No fixation_binary_vectors.pkl found at {path}")


    def generate_saccade_binary_vectors(self, return_df: bool = False) -> Optional[pd.DataFrame]:
        """
        Generates binary vectors for each saccade 'from' and 'to' category per run and agent.
        Stores the result in self.saccade_binary_vectors and optionally returns it.

        Args:
            return_df (bool): If True, returns the binary vector DataFrame.

        Returns:
            Optional[pd.DataFrame]: Long-format DataFrame of binary vectors (if return_df=True).
        """
        logger.info("\n ** Generating saccade binary vectors...")

        if self.saccades is None or self.saccades.empty:
            logger.info("Saccade dataframe not loaded yet. Attempting to load from disk.")
            self.load_dataframes("saccades")

        if not {"from", "to", "start", "stop"}.issubset(self.saccades.columns):
            logger.error("Saccade dataframe must contain 'from', 'to', 'start', and 'stop' columns.")
            return None

        run_lengths_df = self.gaze_data.get_data('run_lengths')
        vectors = []

        grouped = self.saccades.groupby(["session_name", "run_number", "agent"])
        for (session, run, agent), group in grouped:
            run_len_match = run_lengths_df.query("session_name == @session and run_number == @run")
            if run_len_match.empty:
                logger.warning("Run length missing for %s-%s-%s — skipping", session, run, agent)
                continue
            run_length = int(run_len_match["run_length"].values[0])

            from_dict = defaultdict(lambda: np.zeros(run_length, dtype=int))
            to_dict = defaultdict(lambda: np.zeros(run_length, dtype=int))

            for _, row in group.iterrows():
                start, stop = row["start"], row["stop"]
                from_cat, to_cat = row["from"], row["to"]
                from_dict[from_cat][start:stop + 1] = 1
                to_dict[to_cat][start:stop + 1] = 1

            for cat, vec in from_dict.items():
                vectors.append({
                    "session_name": session,
                    "run_number": run,
                    "agent": agent,
                    "from_or_to": "from",
                    "saccade_type": cat,
                    "binary_vector": vec
                })

            for cat, vec in to_dict.items():
                vectors.append({
                    "session_name": session,
                    "run_number": run,
                    "agent": agent,
                    "from_or_to": "to",
                    "saccade_type": cat,
                    "binary_vector": vec
                })

        result_df = pd.DataFrame(vectors)
        self.saccade_binary_vectors = result_df
        logger.info("Saccade binary vector generation complete.")
        if return_df:
            return result_df

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
