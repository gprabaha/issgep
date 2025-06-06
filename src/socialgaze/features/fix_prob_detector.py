# src/socialgaze/features/fix_prob_detector.py

import pdb
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
        allowed_modes =self.config.modes
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
                periods = [("all", [], [(0, run_length - 1)])]
            elif mode == "interactivity":
                int_df = interactivity_df.query(
                    "session_name == @session and run_number == @run"
                )
                int_periods = list(zip(int_df["start"], int_df["stop"]))
                nonint_periods = self._invert_periods(int_periods, run_length)
                periods = [("interactive", [], int_periods), ("non_interactive", [], nonint_periods)]
            elif mode == "segments":
                int_df = interactivity_df.query(
                    "session_name == @session and run_number == @run"
                )
                int_periods = list(zip(int_df["start"], int_df["stop"]))
                nonint_periods = self._invert_periods(int_periods, run_length)
                periods = [(f"{label}", f"{i}", [(s, e)])
                           for label, plist in [("interactive", int_periods), ("non_interactive", nonint_periods)]
                           for i, (s, e) in enumerate(plist)]

            for label, index, period_list in periods:
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
                        row["segment_id"] = int(index)
                        row["start"], row["stop"] = period_list[0]

                    joint_probs.append(row)

        df = pd.DataFrame(joint_probs)
        self._save_output(df, mode)
        return df


    def _save_output(self, df: pd.DataFrame, mode: str):
        allowed_modes =self.config.modes
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
        path_map = {
            "overall": self.config.fix_prob_df_path,
            "interactivity": self.config.fix_prob_df_by_interactivity_path,
            "segments": self.config.fix_prob_df_by_interactivity_segment_path
        }
        allowed_modes =self.config.modes
        if mode is None:
            logger.info(f"No mode provided to _load_output. Loading all modes")
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


import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from tqdm import tqdm

def estimate_probabilities_and_errors(a: np.ndarray, b: np.ndarray):
    """Estimate P(A), P(B), P(AB) with standard errors accounting for autocorrelation and cross-correlation."""
    assert len(a) == len(b)
    N = len(a)
    a = a.astype(bool)
    b = b.astype(bool)
    ab = a & b

    # Means
    p_a = np.mean(a)
    p_b = np.mean(b)
    p_ab = np.mean(ab)

    # Lag-1 autocorrelations
    r_a = acf(a, nlags=1, fft=False)[1]
    r_b = acf(b, nlags=1, fft=False)[1]
    r_ab = acf(ab, nlags=1, fft=False)[1]

    # Effective sample sizes
    N_eff_a = N * (1 - r_a) / (1 + r_a)
    N_eff_b = N * (1 - r_b) / (1 + r_b)
    N_eff_ab = N * (1 - r_ab) / (1 + r_ab)

    # Bound check
    N_eff_a = max(1, N_eff_a)
    N_eff_b = max(1, N_eff_b)
    N_eff_ab = max(1, N_eff_ab)

    # Standard errors
    se_a = np.sqrt(p_a * (1 - p_a) / N_eff_a)
    se_b = np.sqrt(p_b * (1 - p_b) / N_eff_b)
    se_ab = np.sqrt(p_ab * (1 - p_ab) / N_eff_ab)

    return {
        "P(A)": (p_a, se_a),
        "P(B)": (p_b, se_b),
        "P(AB)": (p_ab, se_ab)
    }

def permutation_test_pab(a, b, n_permutations=1000):
    """Permutation test comparing P(AB) to expected P(A)*P(B)"""
    a = a.astype(bool)
    b = b.astype(bool)
    observed_pab = np.mean(a & b)
    pab_shuffled = []

    for _ in range(n_permutations):
        b_shifted = np.roll(b, np.random.randint(1, len(b)))
        pab_shuffled.append(np.mean(a & b_shifted))

    pab_shuffled = np.array(pab_shuffled)
    p_value = np.mean(pab_shuffled >= observed_pab)

    return observed_pab, pab_shuffled, p_value

def sliding_window_dependence(a, b, window_size=1000, step_size=100):
    """Sliding window estimation of P(A), P(B), P(AB), and independence deviation."""
    assert len(a) == len(b)
    indices = range(0, len(a) - window_size + 1, step_size)
    results = []

    for start in indices:
        end = start + window_size
        aw = a[start:end]
        bw = b[start:end]

        p_a = np.mean(aw)
        p_b = np.mean(bw)
        p_ab = np.mean(aw & bw)
        delta = p_ab - (p_a * p_b)

        results.append((start, p_a, p_b, p_ab, delta))

    return np.array(results)

def gmm_on_dependence_deltas(delta_array, n_components=2):
    """Fit GMM to deltas to cluster states of dependence"""
    delta_array = delta_array.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(delta_array)
    labels = gmm.predict(delta_array)
    return gmm, labels

def fit_hmm_to_joint_state(a, b, n_states=2):
    """Fit a discrete HMM to the joint state of A and B"""
    joint_states = (a.astype(int) << 1) | b.astype(int)  # 2-bit state: 00, 01, 10, 11 -> 0, 1, 2, 3
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, random_state=0)
    model.fit(joint_states.reshape(-1, 1))
    state_sequence = model.predict(joint_states.reshape(-1, 1))
    return model, state_sequence

# Example usage (a and b must be defined as binary arrays of same length)
# a, b = <binary numpy arrays>
# results = estimate_probabilities_and_errors(a, b)
# observed_pab, pab_shuffled, p_val = permutation_test_pab(a, b)
# sliding_results = sliding_window_dependence(a, b)
# gmm, gmm_labels = gmm_on_dependence_deltas(sliding_results[:, 4])
# hmm_model, hmm_states = fit_hmm_to_joint_state(a, b)
