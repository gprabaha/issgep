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
    def __init__(self, fixation_detector, config, interactivity_detector=None):
        """
        Args:
            fixation_detector (FixationDetector): Initialized detector with fixation data.
            config (FixationProbabilityConfig): Config with save/load path info.
            interactivity_detector (Optional[InteractivityDetector]): For separating interactive/non-interactive calculations.
        """
        self.detector = fixation_detector
        self.config = config
        self.interactivity_detector = interactivity_detector
        self.fixation_prob_df: Optional[pd.DataFrame] = None
        self.fixation_prob_df_by_interactivity: Optional[pd.DataFrame] = None

    def compute_and_save(self) -> pd.DataFrame:
        logger.info("Computing overall fixation probabilities")
        joint_probs = []

        if self.detector.fixations is None:
            logger.info("Fixations not loaded, using get_behavior_data to populate them.")
            self.detector.fixations = self.detector.get_behavior_data("fixations")

        if self.detector.gaze_data.run_lengths is None:
            logger.info("Run lengths not loaded, using get_data('run_lengths') to populate them.")
            self.detector.gaze_data.run_lengths = self.detector.gaze_data.get_data("run_lengths")

        fixation_df = self.detector.fixations
        run_lengths_df = self.detector.gaze_data.run_lengths
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

            available_categories = pd.concat([m1_df["category"], m2_df["category"]]).unique()
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
        logger.info(f"Saved overall fixation probabilities to {save_path}")
        return self.fixation_prob_df

    def compute_by_interactivity_and_save(self) -> pd.DataFrame:
        logger.info("Computing fixation probabilities split by interactivity")
        joint_probs = []

        if self.detector.fixations is None:
            logger.info("Fixations not loaded, using get_behavior_data to populate them.")
            self.detector.fixations = self.detector.get_behavior_data("fixations")

        if self.detector.gaze_data.run_lengths is None:
            logger.info("Run lengths not loaded, using get_data('run_lengths') to populate them.")
            self.detector.gaze_data.run_lengths = self.detector.gaze_data.get_data("run_lengths")

        if self.interactivity_detector is None:
            raise ValueError("InteractivityDetector is required for interactivity-based fixation probability analysis.")

        fixation_df = self.detector.fixations
        run_lengths_df = self.detector.gaze_data.run_lengths
        interactivity_df = self.interactivity_detector.get_interactivity_periods()

        grouped = fixation_df.groupby(["session_name", "run_number"])

        for (session, run), sub_df in tqdm(grouped, desc="Processing run with interactivity"):
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

            # Get interactivity periods
            int_df = interactivity_df[
                (interactivity_df["session_name"] == session) &
                (interactivity_df["run_number"] == run)
            ]
            int_periods = list(zip(int_df["start"], int_df["stop"]))
            nonint_periods = self._invert_periods(int_periods, run_length)

            for period_label, period_list in [("interactive", int_periods), ("non_interactive", nonint_periods)]:
                total_duration = sum(end - start + 1 for start, end in period_list)
                if total_duration == 0:
                    continue

                m1_filtered = self._restrict_to_periods(m1_df, period_list)
                m2_filtered = self._restrict_to_periods(m2_df, period_list)

                available_categories = pd.concat([m1_filtered["category"], m2_filtered["category"]]).unique()
                for category in sorted(available_categories):
                    m1_subset = m1_filtered[m1_filtered["category"] == category]
                    m2_subset = m2_filtered[m2_filtered["category"] == category]

                    m1_indices = list(zip(m1_subset["start"], m1_subset["stop"]))
                    m2_indices = list(zip(m2_subset["start"], m2_subset["stop"]))

                    joint_duration = self._compute_joint_duration(m1_indices, m2_indices)
                    p_m1 = sum(stop - start + 1 for start, stop in m1_indices) / total_duration
                    p_m2 = sum(stop - start + 1 for start, stop in m2_indices) / total_duration
                    p_joint = joint_duration / total_duration

                    joint_probs.append({
                        "monkey_pair": f"{m1}-{m2}",
                        "session_name": session,
                        "run_number": run,
                        "interactivity": period_label,
                        "fixation_category": category,
                        "P(m1)": p_m1,
                        "P(m2)": p_m2,
                        "P(m1)*P(m2)": p_m1 * p_m2,
                        "P(m1&m2)": p_joint
                    })

        self.fixation_prob_df_by_interactivity = pd.DataFrame(joint_probs)
        save_path = self.config.fix_prob_df_by_interactivity_path
        save_df_to_pkl(self.fixation_prob_df_by_interactivity, save_path)
        logger.info(f"Saved fixation probabilities by interactivity to {save_path}")
        return self.fixation_prob_df_by_interactivity


    def get_data(self) -> pd.DataFrame:
        load_path = self.config.fix_prob_df_path
        logger.info(f"Loading fixation probability dataframe from {load_path}")
        self.fixation_prob_df = load_df_from_pkl(load_path)
        return self.fixation_prob_df

    def get_data_by_interactivity(self) -> pd.DataFrame:
        load_path = self.config.fix_prob_df_by_interactivity_path
        logger.info(f"Loading interactivity-based fixation probability dataframe from {load_path}")
        self.fixation_prob_df_by_interactivity = load_df_from_pkl(load_path)
        return self.fixation_prob_df_by_interactivity

    def _compute_joint_duration(self, m1_ranges: List[Tuple[int, int]], m2_ranges: List[Tuple[int, int]]) -> int:
        joint = 0
        for s1, e1 in m1_ranges:
            for s2, e2 in m2_ranges:
                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)
                if overlap_end >= overlap_start:
                    joint += (overlap_end - overlap_start + 1)
        return joint


    def _restrict_to_periods(self, df: pd.DataFrame, periods: List[Tuple[int, int]]) -> pd.DataFrame:
        """Return rows that overlap with any period and clip their start/stop within bounds."""
        result = []
        for start, stop in periods:
            sub = df[(df["stop"] >= start) & (df["start"] <= stop)].copy()
            sub["start"] = sub["start"].clip(lower=start)
            sub["stop"] = sub["stop"].clip(upper=stop)
            result.append(sub)
        return pd.concat(result) if result else pd.DataFrame(columns=df.columns)


    def _invert_periods(self, intervals: List[Tuple[int, int]], max_val: int) -> List[Tuple[int, int]]:
        if not intervals:
            return [(0, max_val - 1)]
        intervals = sorted(intervals)
        result = []
        prev_end = 0
        for start, end in intervals:
            if start > prev_end:
                result.append((prev_end, start - 1))
            prev_end = max(prev_end, end + 1)
        if prev_end <= max_val - 1:
            result.append((prev_end, max_val - 1))
        return result
