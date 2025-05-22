# src/socialgaze/features/fix_prob_detector.py

import pdb
import logging
from typing import Optional, List, Tuple
from tqdm import tqdm
import pandas as pd

from socialgaze.utils.saving_utils import save_df_to_pkl 
from socialgaze.utils.loading_utils import load_df_from_pkl

logger = logging.getLogger(__name__)


class FixProbDetector:
    def __init__(self, fixation_detector, config):
        """
        Args:
            fixation_detector (FixationDetector): Initialized detector with fixation data.
            config (FixationProbabilityConfig): Config with save/load path info.
        """
        self.detector = fixation_detector
        self.config = config
        self.fixation_prob_df: Optional[pd.DataFrame] = None


    def compute_and_save(self) -> pd.DataFrame:
        logger.info("Computing fixation statistics")
        joint_probs = []

        if self.detector.fixations is None:
            logger.info("Fixations not loaded, using get_behavior_data to populate them.")
            self.detector.fixations = self.detector.get_behavior_data("fixations")

        if self.detector.gaze_data.run_lengths is None:
            logger.info("Run lengths not loaded, using get_data('run_lengths') to populate them.")
            self.detector.gaze_data.run_lengths = self.detector.gaze_data.get_data("run_lengths")

        fixation_df = self.detector.fixations
        run_lengths_df = self.detector.gaze_data.run_lengths  # Expected to have session_name, run_number, run_length columns

        grouped = fixation_df.groupby(["session_name", "run_number"])

        for (session, run), sub_df in tqdm(grouped, desc="Processing run"):
            try:
                row = self.detector.config.ephys_days_and_monkeys_df
                session_row = row[row["session_name"] == session].iloc[0]
                m1 = session_row["m1"]
                m2 = session_row["m2"]
            except IndexError:
                logger.warning(f"Session {session} not found in ephys_days_and_monkeys_df.")
                continue

            m1_df = sub_df[sub_df["agent"] == "m1"]
            m2_df = sub_df[sub_df["agent"] == "m2"]

            if m1_df.empty or m2_df.empty:
                continue

            try:
                run_length = run_lengths_df[
                    (run_lengths_df["session_name"] == session) &
                    (run_lengths_df["run_number"] == run)
                ]["run_length"].values[0]
            except IndexError:
                logger.warning(f"Run length for session {session}, run {run} not found in DataFrame.")
                continue

            available_categories = sub_df["category"].unique()
            for category in sorted(available_categories):
                m1_subset = m1_df[m1_df["category"] == category]
                m2_subset = m2_df[m2_df["category"] == category]

                m1_indices = list(zip(m1_subset["start"], m1_subset["stop"]))
                m2_indices = list(zip(m2_subset["start"], m2_subset["stop"]))

                joint_duration = self._compute_joint_duration(m1_indices, m2_indices)
                p_m1 = sum(stop - start + 1 for start, stop in m1_indices) / run_length
                p_m2 = sum(stop - start + 1 for start, stop in m2_indices) / run_length
                p_joint = joint_duration / run_length

                joint_probs.append({
                    "monkey_pair": f"{m1}-{m2}",
                    "session_name": session,
                    "run_number": run,
                    "fixation_category": category,
                    "P(m1)": p_m1,
                    "P(m2)": p_m2,
                    "P(m1)*P(m2)": p_m1 * p_m2,
                    "P(m1&m2)": p_joint
                })

        self.fixation_prob_df = pd.DataFrame(joint_probs)
        save_path = self.config.fix_prob_df_path
        save_df_to_pkl(self.fixation_prob_df, save_path)
        logger.info(f"Saved fixation probabilities to {save_path}")
        return self.fixation_prob_df


    def get_data(self) -> pd.DataFrame:
        load_path = self.config.fix_prob_df_path
        logger.info(f"Loading fixation probability dataframe from {load_path}")
        self.fixation_prob_df = load_df_from_pkl(load_path)
        return self.fixation_prob_df


    def _compute_joint_duration(self, m1_ranges: List[Tuple[int, int]], m2_ranges: List[Tuple[int, int]]) -> int:
        joint = 0
        for s1, e1 in m1_ranges:
            for s2, e2 in m2_ranges:
                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)
                if overlap_end >= overlap_start:
                    joint += (overlap_end - overlap_start + 1)
        return joint