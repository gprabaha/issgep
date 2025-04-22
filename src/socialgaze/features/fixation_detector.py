# src/socialgaze/features/fixation_detector.py

import pdb

import logging
from typing import List, Tuple, Optional
from pathlib import Path
import random
from multiprocessing import Pool
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
        self.fixations = pd.DataFrame()
        self.saccades = pd.DataFrame()
        self.fixation_binary_vectors = pd.DataFrame()

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
        grouped = self.fixations.groupby(["session_name", "run_number", "agent"])
        log_interval = getattr(self.config, "fixation_labeling_log_interval", 100)

        for idx, ((session, run, agent), group) in enumerate(grouped):
            if idx % log_interval == 0:
                logger.info("Processing fixation group %d / %d", idx, len(grouped))
            self._annotate_fixation_group_with_rois(group, session, run, agent)


    def _annotate_fixation_group_with_rois(self, group, session, run, agent):
        if len(group) != 1:
            logger.warning("Expected 1 row for %s-%s-%s but found %d", session, run, agent, len(group))
            return

        pos_df = self.gaze_data.get_data("positions", session_name=session, run_number=run, agent=agent)
        roi_df = self.gaze_data.get_data("roi_vertices", session_name=session, run_number=run, agent=agent)

        if pos_df is None or pos_df.empty or roi_df is None or roi_df.empty:
            logger.warning("Missing gaze or ROI data for %s-%s-%s", session, run, agent)
            return

        x = np.array(pos_df.iloc[0]["x"])
        y = np.array(pos_df.iloc[0]["y"])
        roi_rects = roi_df[["roi_name", "x_min", "y_min", "x_max", "y_max"]]

        row = group.iloc[0]
        starts = row["starts"]
        stops = row["stops"]
        locations = []

        for start, stop in zip(starts, stops):
            mean_x = np.mean(x[start:stop + 1])
            mean_y = np.mean(y[start:stop + 1])
            location = _find_matching_rois((mean_x, mean_y), roi_rects)
            locations.append(location)

        self.fixations.at[group.index[0], "location"] = locations


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
        grouped = self.saccades.groupby(["session_name", "run_number", "agent"])
        log_interval = getattr(self.config, "fixation_labeling_log_interval", 100)

        for idx, ((session, run, agent), group) in enumerate(grouped):
            if idx % log_interval == 0:
                logger.info("Processing saccade group %d / %d", idx, len(grouped))
            self._annotate_saccade_group_with_rois(group, session, run, agent)

    
    def _annotate_saccade_group_with_rois(self, group, session, run, agent):
        if len(group) != 1:
            logger.warning("Expected 1 row for %s-%s-%s but found %d", session, run, agent, len(group))
            return

        pos_df = self.gaze_data.get_data("positions", session_name=session, run_number=run, agent=agent)
        roi_df = self.gaze_data.get_data("roi_vertices", session_name=session, run_number=run, agent=agent)

        if pos_df is None or pos_df.empty or roi_df is None or roi_df.empty:
            logger.warning("Missing gaze or ROI data for %s-%s-%s", session, run, agent)
            return

        x = np.array(pos_df.iloc[0]["x"])
        y = np.array(pos_df.iloc[0]["y"])
        roi_rects = roi_df[["roi_name", "x_min", "y_min", "x_max", "y_max"]]

        row = group.iloc[0]
        starts = row["starts"]
        stops = row["stops"]
        from_rois = []
        to_rois = []

        for start, stop in zip(starts, stops):
            from_pos = (x[start], y[start])
            to_pos = (x[stop], y[stop])
            from_rois.append(_find_matching_rois(from_pos, roi_rects))
            to_rois.append(_find_matching_rois(to_pos, roi_rects))

        self.saccades.at[group.index[0], "from"] = from_rois
        self.saccades.at[group.index[0], "to"] = to_rois


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
            
            # Keep reconciling until no changes are made
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

        run_lengths_df = self.gaze_data.get_run_lengths()
        vectors = []

        grouped = self.fixations.groupby(["session_name", "run_number", "agent"])
        for (session, run, agent), group in grouped:
            row = group.iloc[0]
            starts = row["starts"]
            stops = row["stops"]
            categories = row["category"]

            run_len_match = run_lengths_df.query("session_name == @session and run_number == @run")
            if run_len_match.empty:
                logger.warning("Run length missing for %s-%s-%s — skipping", session, run, agent)
                continue
            run_length = int(run_len_match["run_length"].values[0])

            unique_categories = sorted(set(categories))
            binary_dict = {cat: np.zeros(run_length, dtype=int) for cat in unique_categories}

            for (start, stop), cat in zip(zip(starts, stops), categories):
                binary_dict[cat][start:stop + 1] = 1

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

        fixation_row = fixation_groups.get_group(key).iloc[0]
        saccade_row = saccade_groups.get_group(key).iloc[0]

        fixation_starts = fixation_row["starts"]
        fixation_stops = fixation_row["stops"]
        fixation_locs = fixation_row["location"].copy()

        saccade_starts = saccade_row["starts"]
        saccade_stops = saccade_row["stops"]
        saccade_froms = saccade_row["from"].copy()
        saccade_tos = saccade_row["to"].copy()

        events = _merge_and_sort_gaze_events(fixation_starts, fixation_stops, saccade_starts, saccade_stops)

        new_fix_locs, new_sacc_froms, new_sacc_tos, changes_made = self._correct_event_label_mismatches(
            session_name, run_number, agent, events, fixation_locs, saccade_froms, saccade_tos
        )
        self.fixations.at[fixation_row.name, "location"] = new_fix_locs
        self.saccades.at[saccade_row.name, "from"] = new_sacc_froms
        self.saccades.at[saccade_row.name, "to"] = new_sacc_tos
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
        "session_name": [session_name],
        "run_number": [run_number],
        "agent": [agent],
        "starts": [events[:, 0]],
        "stops": [events[:, 1]]
    })


# Fixation and saccade labeling functions

def _find_matching_rois(position: np.ndarray, roi_df: pd.DataFrame) -> List[str]:
    matching_rois = []
    for _, roi in roi_df.iterrows():
        if roi["x_min"] <= position[0] <= roi["x_max"] and roi["y_min"] <= position[1] <= roi["y_max"]:
            matching_rois.append(roi["roi_name"])
    return matching_rois if matching_rois else ["out_of_roi"]

def _merge_and_sort_gaze_events(fix_starts, fix_stops, sacc_starts, sacc_stops):
    events = [(s, e, "fixation", i) for i, (s, e) in enumerate(zip(fix_starts, fix_stops))]
    events += [(s, e, "saccade", i) for i, (s, e) in enumerate(zip(sacc_starts, sacc_stops))]
    events.sort(key=lambda tup: tup[0])
    return events


def _categorize_fixations(location_list):
    """
    Takes a list of ROI lists (one per fixation) and returns a list of single-category labels
    ('face', 'object', or 'out_of_roi') for each fixation, mutually exclusive.
    """
    face_rois = {"face", "mouth", "eyes", "eyes_nf"}
    object_rois = {"left_nonsocial_object", "right_nonsocial_object"}
    categorized = []
    for roi_list in location_list:
        roi_set = set(roi_list)
        if roi_set & face_rois:
            categorized.append("face")
        elif roi_set & object_rois:
            categorized.append("object")
        else:
            categorized.append("out_of_roi")
    return categorized
