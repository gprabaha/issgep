# src/socialgaze/features/fix_prob_detector.py

import logging
from typing import Optional, List, Tuple
from tqdm import tqdm
import pandas as pd

from socialgaze.utils.saving_utils import save_df_to_pkl, load_df_from_pkl

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

        fixation_df = self.detector.fixations
        grouped = fixation_df.groupby(["session_name", "interaction_type", "run_number"])

        for (session, interaction, run), sub_df in tqdm(grouped, desc="Processing sessions"):
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

            m1_fix_locs = self._categorize_fixations(m1_df["fixation_location"].values[0])
            m2_fix_locs = self._categorize_fixations(m2_df["fixation_location"].values[0])
            run_length = m1_df["run_length"].values[0]

            for category in ["eyes", "non_eye_face", "face", "out_of_roi"]:
                if category != "face":
                    m1_indices = [(start, stop) for cat, (start, stop) in zip(m1_fix_locs, m1_df["fixation_start_stop"].values[0]) if cat == category]
                    m2_indices = [(start, stop) for cat, (start, stop) in zip(m2_fix_locs, m2_df["fixation_start_stop"].values[0]) if cat == category]
                else:
                    m1_indices = [(start, stop) for cat, (start, stop) in zip(m1_fix_locs, m1_df["fixation_start_stop"].values[0]) if cat in {"eyes", "non_eye_face"}]
                    m2_indices = [(start, stop) for cat, (start, stop) in zip(m2_fix_locs, m2_df["fixation_start_stop"].values[0]) if cat in {"eyes", "non_eye_face"}]

                joint_duration = self._compute_joint_duration(m1_indices, m2_indices)
                p_m1 = sum(stop + 1 - start for start, stop in m1_indices) / run_length
                p_m2 = sum(stop + 1 - start for start, stop in m2_indices) / run_length
                p_joint = joint_duration / run_length

                joint_probs.append({
                    "monkey_pair": f"{m1}-{m2}",
                    "session_name": session,
                    "interaction_type": interaction,
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


    def _categorize_fixations(self, fix_locations: List[List[str]]) -> List[str]:
        return [
            "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
            "non_eye_face" if set(fixes) & {"mouth", "face"} else
            "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
            for fixes in fix_locations
        ]

    def _compute_joint_duration(self, m1_ranges: List[Tuple[int, int]], m2_ranges: List[Tuple[int, int]]) -> int:
        joint = 0
        for s1, e1 in m1_ranges:
            for s2, e2 in m2_ranges:
                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)
                if overlap_end >= overlap_start:
                    joint += (overlap_end - overlap_start + 1)
        return joint