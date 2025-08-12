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



import os
import logging
from typing import Iterable, Literal, Tuple, Optional, Dict
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats.mstats import winsorize



class FixProbPlotter(FixProbDetector):
    """
    Plotter for fixation probability summaries (joint vs marginal).
    Inherits data access from FixProbDetector.

    Default context: "overall".
    """

    def __init__(self, fixation_detector, config, interactivity_detector=None):
        super().__init__(
            fixation_detector=fixation_detector,
            config=config,
            interactivity_detector=interactivity_detector
        )
        # Headless-safe backend for HPC
        matplotlib.use("Agg", force=True)

    # ------------- Public API ------------- #
    def plot_joint_vs_marginal_violin(
        self,
        context: Literal["overall", "interactivity", "segments"] = "overall",
        categories: Iterable[str] = ("face", "out_of_roi"),
        export_dir: Optional[str] = None,
        filename: Optional[str] = None,
        export_formats: Tuple[str, ...] = ("pdf",),  # add "svg" if you want SVG too
        illustrator_friendly: bool = True,
        return_fig: bool = False,
    ):
        """
        Make Illustrator-friendly violin plots comparing P(m1&m2) vs P(m1)*P(m2).

        Parameters
        ----------
        context : {"overall","interactivity","segments"}
            "overall" (default) for one-row summary; others split by interactivity.
        categories : iterable of {"face","out_of_roi", ...}
        export_dir : path to save; defaults to config.plot_dir / "fix_prob"
        filename : custom base filename; sensible defaults are chosen per context
        export_formats : ("pdf",) or ("pdf","svg")
        illustrator_friendly : keep live text and vectors
        return_fig : if True, returns (fig, axes) instead of closing
        """
        df = self.get_data(context)  # uses the detector’s API
        df = df[df["fixation_category"].isin(categories)].copy()
        
        # Alias columns used downstream
        df["joint"] = df["P(m1&m2)"]
        df["marginal"] = df["P(m1)*P(m2)"]

        # Palettes
        violin_palette = getattr(self.config, "violin_palette", {"P(m1)*P(m2)": "#8da0cb", "P(m1&m2)": "#fc8d62"})
        monkey_pairs = df["monkey_pair"].unique()
        monkey_color_dict = self._get_monkey_palette(monkey_pairs)

        # Figure layouts
        if context == "overall":
            fig, axes = plt.subplots(1, len(categories), figsize=(12, 6), sharey=False)
            axes = axes.ravel().tolist()
            axes = axes if isinstance(axes, (list, tuple, pd.Series)) else [axes]
            for i, category in enumerate(categories):
                ax = axes[i]
                sub_df = df[df["fixation_category"] == category].copy()
                self._plot_violin_core(ax, sub_df, monkey_color_dict, violin_palette)
                ax.set_title(f"{category.capitalize()}")
            fig.suptitle("Fixation Probability Comparison — Overall", fontsize=16)
            default_name = "joint_fixation_probabilities_overall"

        elif context in {"interactivity", "segments"}:
            fig, axs = plt.subplots(2, len(categories), figsize=(12, 10), sharey=False)
            interactivities = ["interactive", "non_interactive"]
            for i, interactivity in enumerate(interactivities):
                for j, category in enumerate(categories):
                    ax = axs[i, j]
                    sub_df = df[
                        (df["interactivity"] == interactivity) &
                        (df["fixation_category"] == category)
                    ].copy()
                    self._plot_violin_core(ax, sub_df, monkey_color_dict, violin_palette)
                    ax.set_title(f"{interactivity.capitalize()} — {category}")
            title = "By Interactivity Segment" if context == "segments" else "By Interactivity"
            fig.suptitle(f"Fixation Probability Comparison — {title}", fontsize=16)
            default_name = f"joint_fixation_probabilities_by_{context}"

        else:
            raise ValueError(f"Invalid context: {context}")

        # Layout + export
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if illustrator_friendly:
            self._set_illustrator_friendly_rcparams()

        out_dir = export_dir or self.config.plot_dir
        os.makedirs(out_dir, exist_ok=True)

        base = filename or default_name
        self._finalize_and_save(fig, out_dir, base, export_formats)

        if return_fig:
            return fig, fig.get_axes()
        plt.close(fig)

    # ------------- Internals ------------- #
    def _plot_violin_core(
        self,
        ax: matplotlib.axes.Axes,
        full_df: pd.DataFrame,
        monkey_color_dict: Dict,
        violin_palette: Dict[str, str]
    ):
        """One violin panel with winsorization + per-pair medians and KS test."""
        if full_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        melted = full_df.melt(
            id_vars=["monkey_pair"],
            value_vars=["P(m1)*P(m2)", "P(m1&m2)"],
            var_name="Probability Type",
            value_name="Probability"
        )

        # Winsorize within each probability type (robust outlier handling)
        melted["Probability"] = (
            melted
            .groupby("Probability Type", group_keys=False)["Probability"]
            .apply(lambda x: pd.Series(winsorize(x, limits=[0.05, 0.05]), index=x.index))
        )

        sns.violinplot(
            data=melted,
            x="Probability Type",
            y="Probability",
            hue="Probability Type",
            palette=violin_palette,
            inner="quartile",
            order=["P(m1)*P(m2)", "P(m1&m2)"],
            legend=False,
            ax=ax
        )

        # Overlay per-monkey-pair medians (two points + a faint connector)
        mp_medians = self._get_monkey_pair_medians(full_df)
        self._overlay_medians(ax, mp_medians, monkey_color_dict)

        # KS annotation
        self._annotate_ks(ax, full_df)

        # Cosmetics
        ax.set_xlabel("")
        ax.set_ylabel("Probability")

    @staticmethod
    def _get_monkey_palette(monkey_pairs):
        palette = sns.color_palette("Set2", n_colors=len(monkey_pairs))
        return {mp: palette[i] for i, mp in enumerate(monkey_pairs)}

    @staticmethod
    def _get_monkey_pair_medians(full_df: pd.DataFrame) -> dict:
        cols = ["P(m1)*P(m2)", "P(m1&m2)"]
        agg_df = full_df.groupby("monkey_pair")[cols].median().reset_index()
        return {row["monkey_pair"]: [row[cols[0]], row[cols[1]]] for _, row in agg_df.iterrows()}

    @staticmethod
    def _overlay_medians(ax, monkey_pair_data, monkey_color_dict):
        for mp, y_vals in monkey_pair_data.items():
            if not isinstance(y_vals, list) or len(y_vals) != 2:
                logger.warning(f"Invalid y_vals for {mp}: {y_vals}")
                continue
            jitter = 0.01 * (hash(mp) % 10 - 5)
            x0, x1 = 0 + jitter, 1 + jitter
            color = monkey_color_dict.get(mp, "gray")
            ax.plot([x0, x1], y_vals, color=color, alpha=0.4, linewidth=1.25)
            ax.scatter([x0], [y_vals[0]], color=color, s=30, alpha=0.8)
            ax.scatter([x1], [y_vals[1]], color=color, s=30, alpha=0.8)

    @staticmethod
    def _annotate_ks(ax, group_df: pd.DataFrame):
        try:
            ks_stat, p_val = ks_2samp(group_df["P(m1)*P(m2)"], group_df["P(m1&m2)"])
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            ax.text(0.5, 1.05, "KS test: N/A", transform=ax.transAxes, ha="center", fontsize=12)
            return

        marker = FixProbPlotter._get_significance_marker(p_val)
        # Put the marker above the axes (kept as text for Illustrator)
        ax.text(0.5, 1.05, f"KS test: {marker}", transform=ax.transAxes, ha="center", fontsize=12)

    @staticmethod
    def _get_significance_marker(p_val: float) -> str:
        if p_val < 1e-3:
            return f'***; p = {p_val:.2e}'
        elif p_val < 1e-2:
            return f'**; p = {p_val:.2e}'
        elif p_val < 5e-2:
            return f'*; p = {p_val:.3f}'
        else:
            return f'NS; p = {p_val:.3f}'

    @staticmethod
    def _set_illustrator_friendly_rcparams():
        """
        Keep text as text, embed TrueType fonts, avoid path-outlining,
        and keep transparency for clean editing in Illustrator.
        """
        # PDF: embed TrueType fonts
        matplotlib.rcParams["pdf.fonttype"] = 42
        # PS (in case): embed TrueType fonts
        matplotlib.rcParams["ps.fonttype"] = 42
        # SVG: keep text as text (not paths)
        matplotlib.rcParams["svg.fonttype"] = "none"
        # No global facecolor fills; export will set transparent=True

    @staticmethod
    def _finalize_and_save(fig, save_dir: str, base_filename: str, formats: Tuple[str, ...]):
        fig.patch.set_alpha(0)
        for ax in fig.get_axes():
            ax.set_facecolor("none")

        for fmt in formats:
            path = os.path.join(save_dir, f"{base_filename}.{fmt.lower()}")
            try:
                fig.savefig(
                    path,
                    format=fmt,
                    transparent=True,
                    bbox_inches="tight",
                    dpi=300 if fmt.lower() in ("png",) else None,
                    metadata={"Creator": "socialgaze FixProbPlotter"}
                )
                logger.info(f"Saved: {path}")
            except Exception as e:
                logger.error(f"Failed to save {path}: {e}")




















































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
