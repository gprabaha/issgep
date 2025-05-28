# src/socialgaze/features/fix_prob_detector.py

import logging
from typing import Optional, List, Tuple
import pandas as pd
from tqdm import tqdm

from socialgaze.utils.saving_utils import save_df_to_pkl
from socialgaze.utils.loading_utils import load_df_from_pkl

logger = logging.getLogger(__name__)


class FixProbDetector:
    def __init__(self, fixation_detector, config, interactivity_detector=None):
        self.detector = fixation_detector
        self.config = config
        self.interactivity_detector = interactivity_detector
        self.fixation_prob_df: Optional[pd.DataFrame] = None
        self.fixation_prob_df_by_interactivity: Optional[pd.DataFrame] = None
        self.fixation_prob_df_by_interactivity_segment: Optional[pd.DataFrame] = None


    def compute_fixation_probabilities(self, mode: str = "overall") -> pd.DataFrame:
        allowed_modes = {"overall", "interactivity", "segments"}
        if mode not in allowed_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {allowed_modes}")

        logger.info(f"Computing fixation probabilities (mode={mode})")
        joint_probs = []

        if self.detector.fixations is None:
            self.detector.fixations = self.detector.get_behavior_data("fixations")

        if self.detector.gaze_data.run_lengths is None:
            self.detector.gaze_data.run_lengths = self.detector.gaze_data.get_data("run_lengths")

        if mode in {"interactivity", "segments"} and self.interactivity_detector is None:
            raise ValueError("InteractivityDetector is required for mode '%s'" % mode)

        fixation_df = self.detector.fixations
        run_lengths_df = self.detector.gaze_data.run_lengths
        grouped = fixation_df.groupby(["session_name", "run_number"])

        if mode in {"interactivity", "segments"}:
            interactivity_df = self.interactivity_detector.get_interactivity_periods()

        for (session, run), sub_df in tqdm(grouped, desc=f"Processing probabilities ({mode})"):
            try:
                session_row = self.detector.config.ephys_days_and_monkeys_df.query(
                    "session_name == @session"
                ).iloc[0]
                m1, m2 = session_row["m1"], session_row["m2"]
            except IndexError:
                logger.warning(f"Session {session} not found in ephys_days_and_monkeys_df.")
                continue

            m1_df = sub_df[sub_df["agent"] == "m1"]
            m2_df = sub_df[sub_df["agent"] == "m2"]
            if m1_df.empty or m2_df.empty:
                continue

            try:
                run_length = run_lengths_df.query(
                    "session_name == @session and run_number == @run"
                )["run_length"].values[0]
            except IndexError:
                logger.warning(f"Run length for session {session}, run {run} not found.")
                continue

            if mode == "overall":
                periods = [("all", [(0, run_length - 1)])]
            elif mode == "interactivity":
                int_df = interactivity_df.query(
                    "session_name == @session and run_number == @run"
                )
                int_periods = list(zip(int_df["start"], int_df["stop"]))
                nonint_periods = self._invert_periods(int_periods, run_length)
                periods = [("interactive", int_periods), ("non_interactive", nonint_periods)]
            elif mode == "segments":
                int_df = interactivity_df.query(
                    "session_name == @session and run_number == @run"
                )
                int_periods = list(zip(int_df["start"], int_df["stop"]))
                nonint_periods = self._invert_periods(int_periods, run_length)
                periods = [(f"{label}_{i}", [(s, e)])
                           for label, plist in [("interactive", int_periods), ("non_interactive", nonint_periods)]
                           for i, (s, e) in enumerate(plist)]

            for label, period_list in periods:
                total_duration = sum(e - s + 1 for s, e in period_list)
                if total_duration == 0:
                    continue

                m1_filt = self._restrict_to_periods(m1_df, period_list)
                m2_filt = self._restrict_to_periods(m2_df, period_list)

                categories = pd.concat([m1_filt["category"], m2_filt["category"]]).unique()
                for cat in sorted(categories):
                    m1_cat = m1_filt[m1_filt["category"] == cat]
                    m2_cat = m2_filt[m2_filt["category"] == cat]

                    m1_idx = list(zip(m1_cat["start"], m1_cat["stop"]))
                    m2_idx = list(zip(m2_cat["start"], m2_cat["stop"]))

                    joint = self._compute_joint_duration(m1_idx, m2_idx)
                    p1 = sum(e - s + 1 for s, e in m1_idx) / total_duration
                    p2 = sum(e - s + 1 for s, e in m2_idx) / total_duration
                    p_joint = joint / total_duration

                    row = {
                        "monkey_pair": f"{m1}-{m2}",
                        "session_name": session,
                        "run_number": run,
                        "fixation_category": cat,
                        "P(m1)": p1,
                        "P(m2)": p2,
                        "P(m1)*P(m2)": p1 * p2,
                        "P(m1&m2)": p_joint
                    }
                    if mode != "overall":
                        row["interactivity"] = label
                    if mode == "segments":
                        row["segment_id"] = int(label.split("_")[-1])
                        row["start"], row["stop"] = period_list[0]

                    joint_probs.append(row)

        df = pd.DataFrame(joint_probs)
        self._save_output(df, mode)
        return df


    def _save_output(self, df: pd.DataFrame, mode: str):
        allowed_modes = {"overall", "interactivity", "segments"}
        if mode not in allowed_modes:
            raise ValueError(f"Invalid mode '{mode}' in _save_output. Must be one of {allowed_modes}")
        
        path_map = {
            "overall": self.config.fix_prob_df_path,
            "interactivity": self.config.fix_prob_df_by_interactivity_path,
            "segments": self.config.fix_prob_df_by_interactivity_segment_path
        }
        save_path = path_map[mode]
        save_df_to_pkl(df, save_path)
        logger.info(f"Saved fixation probabilities ({mode}) to {save_path}")

        if mode == "overall":
            self.fixation_prob_df = df
        elif mode == "interactivity":
            self.fixation_prob_df_by_interactivity = df
        elif mode == "segments":
            self.fixation_prob_df_by_interactivity_segment = df


    def _load_output(self, mode: Optional[str] = None):
        allowed_modes = {"overall", "interactivity", "segments"}
        path_map = {
            "overall": self.config.fix_prob_df_path,
            "interactivity": self.config.fix_prob_df_by_interactivity_path,
            "segments": self.config.fix_prob_df_by_interactivity_segment_path
        }

        if mode is None:
            for m in allowed_modes:
                self._load_output(m)
            return

        if mode not in allowed_modes:
            raise ValueError(f"Invalid mode '{mode}' in _load_output. Must be one of {allowed_modes}")

        path = path_map[mode]
        logger.info(f"Loading fixation probability data from {path}")
        df = load_df_from_pkl(path)

        if mode == "overall":
            self.fixation_prob_df = df
        elif mode == "interactivity":
            self.fixation_prob_df_by_interactivity = df
        elif mode == "segments":
            self.fixation_prob_df_by_interactivity_segment = df



    def get_data(self, mode: Optional[str] = "overall") -> pd.DataFrame | dict:
        """
        Load fixation probability data for a given mode.
        If mode is None, load and return all as a dict.
        """
        if mode is None:
            self._load_output(None)
            return {
                "overall": self.fixation_prob_df,
                "interactivity": self.fixation_prob_df_by_interactivity,
                "segments": self.fixation_prob_df_by_interactivity_segment
            }

        if mode == "overall":
            if self.fixation_prob_df is None:
                self._load_output("overall")
            return self.fixation_prob_df

        elif mode == "interactivity":
            if self.fixation_prob_df_by_interactivity is None:
                self._load_output("interactivity")
            return self.fixation_prob_df_by_interactivity

        elif mode == "segments":
            if self.fixation_prob_df_by_interactivity_segment is None:
                self._load_output("segments")
            return self.fixation_prob_df_by_interactivity_segment

        else:
            raise ValueError(f"Invalid mode '{mode}' in get_data. Must be one of 'overall', 'interactivity', 'segments', or None.")


    def _compute_joint_duration(self, m1_ranges: List[Tuple[int, int]], m2_ranges: List[Tuple[int, int]]) -> int:
        m1_timepoints = set()
        for start, stop in m1_ranges:
            m1_timepoints.update(range(start, stop + 1))

        m2_timepoints = set()
        for start, stop in m2_ranges:
            m2_timepoints.update(range(start, stop + 1))

        joint_timepoints = m1_timepoints & m2_timepoints
        return len(joint_timepoints)


    def _restrict_to_periods(self, df: pd.DataFrame, periods: List[Tuple[int, int]]) -> pd.DataFrame:
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
