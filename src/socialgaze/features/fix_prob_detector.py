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


    def compute_fixation_probabilities(
        self,
        mode: str = "overall",
        category_pairs: list[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        """
        Compute P(m1), P(m2), P(m1)*P(m2), and P(m1&m2) for requested
        agent-category pairings (e.g., m1:object vs m2:face).

        Parameters
        ----------
        mode : {"overall","interactivity","segments"}
            How to segment the timeline.
        category_pairs : list[tuple[str, str]] | None
            List of (m1_category, m2_category) to evaluate. If None, defaults to
            same-category analysis (previous behavior).

        Returns
        -------
        pd.DataFrame
            Columns include:
            - monkey_pair, session_name, run_number
            - m1_category, m2_category, fixation_pair
            - P(m1), P(m2), P(m1)*P(m2), P(m1&m2)
            - interactivity (if mode != "overall")
            - segment_id, start, stop (if mode == "segments")
        """
        allowed_modes = self.config.modes
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
            # Resolve monkey names for nice labeling
            try:
                session_row = self.detector.config.ephys_days_and_monkeys_df.query(
                    "session_name == @session"
                ).iloc[0]
                m1_name, m2_name = session_row["m1"], session_row["m2"]
            except IndexError:
                logger.warning(f"Session {session} not found in ephys_days_and_monkeys_df.")
                continue

            m1_df = sub_df[sub_df["agent"] == "m1"]
            m2_df = sub_df[sub_df["agent"] == "m2"]
            if m1_df.empty or m2_df.empty:
                continue

            # Run length (inclusive indexing downstream)
            try:
                run_length = run_lengths_df.query(
                    "session_name == @session and run_number == @run"
                )["run_length"].values[0]
            except IndexError:
                logger.warning(f"Run length for session {session}, run {run} not found.")
                continue

            # Determine time segments
            if mode == "overall":
                periods = [("all", [], [(0, run_length - 1)])]
            elif mode == "interactivity":
                int_df = interactivity_df.query(
                    "session_name == @session and run_number == @run"
                )
                int_periods = list(zip(int_df["start"], int_df["stop"]))
                nonint_periods = self._invert_periods(int_periods, run_length)
                periods = [
                    ("interactive", [], int_periods),
                    ("non_interactive", [], nonint_periods),
                ]
            elif mode == "segments":
                int_df = interactivity_df.query(
                    "session_name == @session and run_number == @run"
                )
                int_periods = list(zip(int_df["start"], int_df["stop"]))
                nonint_periods = self._invert_periods(int_periods, run_length)
                periods = [
                    (f"{label}", f"{i}", [(s, e)])
                    for label, plist in [("interactive", int_periods), ("non_interactive", nonint_periods)]
                    for i, (s, e) in enumerate(plist)
                ]
            else:
                raise ValueError(f"Unhandled mode: {mode}")

            for label, index, period_list in periods:
                total_duration = sum(e - s + 1 for s, e in period_list)
                if total_duration == 0:
                    continue

                # Restrict fixations to the active period list
                m1_filt = self._restrict_to_periods(m1_df, period_list)
                m2_filt = self._restrict_to_periods(m2_df, period_list)

                # Build default same-category pairs if none provided
                if category_pairs is None:
                    cats = pd.concat([m1_filt["category"], m2_filt["category"]]).dropna().unique()
                    pairs = [(c, c) for c in sorted(cats)]
                else:
                    pairs = category_pairs

                for m1_cat, m2_cat in pairs:
                    # Select events for each agent & category
                    m1_cat_df = m1_filt[m1_filt["category"] == m1_cat]
                    m2_cat_df = m2_filt[m2_filt["category"] == m2_cat]

                    m1_idx = list(zip(m1_cat_df["start"], m1_cat_df["stop"]))
                    m2_idx = list(zip(m2_cat_df["start"], m2_cat_df["stop"]))

                    # Durations
                    dur_m1 = sum(e - s + 1 for s, e in m1_idx)
                    dur_m2 = sum(e - s + 1 for s, e in m2_idx)

                    # Joint overlap (inclusive indices)
                    joint = self._compute_joint_duration(m1_idx, m2_idx)

                    # Probabilities
                    p1 = dur_m1 / total_duration
                    p2 = dur_m2 / total_duration
                    p_joint = joint / total_duration

                    row = {
                        "monkey_pair": f"{m1_name}-{m2_name}",
                        "session_name": session,
                        "run_number": run,
                        "m1_category": m1_cat,
                        "m2_category": m2_cat,
                        "fixation_pair": f"m1_{m1_cat}__vs__m2_{m2_cat}",
                        "P(m1)": p1,
                        "P(m2)": p2,
                        "P(m1)*P(m2)": p1 * p2,
                        "P(m1&m2)": p_joint,
                    }

                    # Backward-compat convenience: only set fixation_category when same
                    if m1_cat == m2_cat:
                        row["fixation_category"] = m1_cat

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
from scipy.stats.mstats import winsorize
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from scipy.stats import ks_2samp, ttest_rel, wilcoxon, binomtest



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
        export_dir: Optional[str] = None,
        filename: Optional[str] = None,
        export_formats: Tuple[str, ...] = ("pdf",),
        illustrator_friendly: bool = True,
        return_fig: bool = False,
    ):
        """
        Violin plots comparing P(m1&m2) vs P(m1)*P(m2) for two cross-agent pairs:
        - m1_face__vs__m2_face
        - m1_object__vs__m2_face

        Assumes probability dataframes were created using:
            pairs=[("face","face"), ("object","face")]
        for all modes.
        """

        df = self.get_data(context).copy()

        required_cols = {"P(m1&m2)", "P(m1)*P(m2)"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure we have pair identifiers (added by the updated compute method)
        if "fixation_pair" not in df.columns:
            if {"m1_category", "m2_category"}.issubset(df.columns):
                df["fixation_pair"] = "m1_" + df["m1_category"].astype(str) + "__vs__m2_" + df["m2_category"].astype(str)
            else:
                raise ValueError("Expected 'fixation_pair' or ('m1_category','m2_category') in dataframe.")

        # Restrict to the two expected pairs; keep deterministic plotting order
        desired_pairs = [
            "m1_face__vs__m2_face",
            "m1_object__vs__m2_face",
        ]
        df = df[df["fixation_pair"].isin(desired_pairs)].copy()
        if df.empty:
            raise ValueError("No rows found for the expected pairs: face-face and object-face.")

        # Working columns
        df["joint"] = df["P(m1&m2)"]
        df["marginal"] = df["P(m1)*P(m2)"]

        # Palettes
        violin_palette = getattr(
            self.config,
            "violin_palette",
            {"P(m1)*P(m2)": "#8da0cb", "P(m1&m2)": "#fc8d62"},
        )
        monkey_pairs = df["monkey_pair"].unique()
        monkey_color_dict = self._get_monkey_palette(monkey_pairs)

        # Column labels
        pretty_label = {
            "m1_face__vs__m2_face": "m1: Face  vs  m2: Face",
            "m1_object__vs__m2_face": "m1: Object  vs  m2: Face",
        }

        def _sub_df_for(pair_key: str, interactivity: str | None = None) -> pd.DataFrame:
            sub = df[df["fixation_pair"] == pair_key]
            if interactivity is not None and "interactivity" in sub.columns:
                sub = sub[sub["interactivity"] == interactivity]
            return sub

        # Build figure(s)
        if context == "overall":
            ncols = 2
            fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6), sharey=False)
            axes = np.atleast_1d(axes).ravel().tolist()

            for col_idx, pair_key in enumerate(desired_pairs):
                ax = axes[col_idx]
                sub_df = _sub_df_for(pair_key).copy()
                self._plot_violin_core(ax, sub_df, monkey_color_dict, violin_palette)
                ax.set_title(pretty_label[pair_key])

            fig.suptitle("Fixation Probability Comparison — Overall", fontsize=16)
            default_name = "joint_fixation_probabilities_pairs_overall"

        elif context in {"interactivity", "segments"}:
            interactivities = ["interactive", "non_interactive"]
            ncols = 2
            fig, axs = plt.subplots(2, ncols, figsize=(6 * ncols, 10), sharey=False)

            for i, interactivity in enumerate(interactivities):
                for j, pair_key in enumerate(desired_pairs):
                    ax = axs[i, j]
                    sub_df = _sub_df_for(pair_key, interactivity).copy()
                    self._plot_violin_core(ax, sub_df, monkey_color_dict, violin_palette)

                    # Column labels on top row only
                    if i == 0:
                        ax.set_title(pretty_label[pair_key])
                    # Row labels on left only
                    if j == 0:
                        ax.set_ylabel(interactivity.capitalize())

            title = "By Interactivity Segment" if context == "segments" else "By Interactivity"
            fig.suptitle(f"Fixation Probability Comparison — {title}", fontsize=16)
            default_name = f"joint_fixation_probabilities_pairs_by_{context}"

        else:
            raise ValueError(f"Invalid context: {context}")

        # Clean axis labels (reduce clutter)
        for ax in fig.get_axes():
            ax.set_xlabel("Monkey Pair")
            ax.set_ylabel("Probability")

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if illustrator_friendly:
            self._set_illustrator_friendly_rcparams()

        out_dir = export_dir or (Path(self.config.plot_dir) / "fix_prob")
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
        self._annotate_stats(ax, full_df)

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
    def _get_significance_marker(p: float) -> str:
        # keep your existing implementation if you already have one
        if p < 1e-4: return "****"
        if p < 1e-3: return "***"
        if p < 1e-2: return "**"
        if p < 5e-2: return "*"
        return "n.s."

    @staticmethod
    def _cohens_d_paired(diffs: np.ndarray) -> float:
        diffs = np.asarray(diffs, dtype=float)
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size < 2:
            return np.nan
        sd = np.std(diffs, ddof=1)
        return np.nan if sd == 0 else float(np.mean(diffs) / sd)

    @staticmethod
    def _rank_biserial_from_wilcoxon(statistic: float, n_eff: int) -> float:
        """
        r_rb = (T+ - T-) / (T+ + T-) = 2*T+ / total_rank_sum - 1,
        where total_rank_sum = n_eff*(n_eff+1)/2 and T+ is the Wilcoxon
        sum of positive ranks (scipy's `wilcoxon` returns T+).
        """
        if n_eff <= 0:
            return np.nan
        total_rank_sum = n_eff * (n_eff + 1) / 2.0
        return 2.0 * float(statistic) / total_rank_sum - 1.0

    @staticmethod
    def _annotate_stats(
        ax,
        group_df: pd.DataFrame,
        *,
        y0: float = 1.08,
        dy: float = 0.055,
        ks_only: bool = False,
    ) -> None:
        """
        Compare distributions of P(m1&m2) vs P(m1)*P(m2) and annotate results.
        Stacks lines above the axes (good for Illustrator).

        Parameters
        ----------
        y0 : top line y (axes coords), dy : vertical spacing, ks_only : if True, only KS line.
        """
        try:
            a = group_df["P(m1&m2)"].to_numpy(dtype=float)
            b = group_df["P(m1)*P(m2)"].to_numpy(dtype=float)
        except KeyError as e:
            ax.text(0.5, y0, f"Missing column: {e}", transform=ax.transAxes, ha="center", fontsize=10)
            return

        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if a.size < 2 or b.size < 2:
            ax.text(0.5, y0, "N too small for tests", transform=ax.transAxes, ha="center", fontsize=10)
            return

        # 1) KS test (distributional difference; not paired)
        try:
            ks_stat, ks_p = ks_2samp(a, b, alternative="two-sided", mode="auto")
            ax.text(
                0.5, y0,
                f"KS: {FixProbPlotter._get_significance_marker(ks_p)} (p={ks_p:.3g})",
                transform=ax.transAxes, ha="center", fontsize=10
            )
        except Exception as e:
            ax.text(0.5, y0, f"KS: N/A ({e})", transform=ax.transAxes, ha="center", fontsize=10)

        if ks_only:
            return

        diffs = a - b

        # 2) Paired t-test + Cohen's dz
        try:
            tres = ttest_rel(a, b, alternative="two-sided", nan_policy="omit")
            t_p = float(tres.pvalue)
            d = FixProbPlotter._cohens_d_paired(diffs)
            ax.text(
                0.5, y0 + dy,
                f"t (paired): {FixProbPlotter._get_significance_marker(t_p)} (p={t_p:.3g}, d={d:.2f})",
                transform=ax.transAxes, ha="center", fontsize=10
            )
        except Exception as e:
            ax.text(0.5, y0 + dy, f"t (paired): N/A ({e})", transform=ax.transAxes, ha="center", fontsize=10)

        # 3) Wilcoxon signed-rank + rank-biserial r
        try:
            # Wilcoxon drops zeros by default with zero_method="wilcox"
            w_stat, w_p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", correction=False)
            n_eff = int(np.sum(diffs != 0))
            r_rb = FixProbPlotter._rank_biserial_from_wilcoxon(w_stat, n_eff)
            ax.text(
                0.5, y0 + 2*dy,
                f"Wilcoxon: {FixProbPlotter._get_significance_marker(w_p)} (p={w_p:.3g}, r={r_rb:.2f})",
                transform=ax.transAxes, ha="center", fontsize=10
            )
        except Exception as e:
            ax.text(0.5, y0 + 2*dy, f"Wilcoxon: N/A ({e})", transform=ax.transAxes, ha="center", fontsize=10)

        # 4) Sign test (binomial test on direction of diffs)
        try:
            npos = int(np.sum(diffs > 0))
            nneg = int(np.sum(diffs < 0))
            n = npos + nneg
            if n == 0:
                raise ValueError("All paired differences are zero.")
            bt = binomtest(k=min(npos, nneg), n=n, p=0.5, alternative="two-sided")
            p_sign = float(bt.pvalue)
            ax.text(
                0.5, y0 + 3*dy,
                f"Sign: {FixProbPlotter._get_significance_marker(p_sign)} (p={p_sign:.3g}, n={n})",
                transform=ax.transAxes, ha="center", fontsize=10
            )
        except Exception as e:
            ax.text(0.5, y0 + 3*dy, f"Sign: N/A ({e})", transform=ax.transAxes, ha="center", fontsize=10)

    # --- Backward-compat wrapper (keeps the same signature you had) ---
    @staticmethod
    def _annotate_ks(ax, group_df: pd.DataFrame) -> None:
        # Delegate to the multi-test annotator but only print the KS line
        FixProbPlotter._annotate_stats(ax, group_df, y0=1.05, dy=0.06, ks_only=True)


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
        matplotlib.rcParams.update({
            "pdf.fonttype": 42,          # embed TrueType
            "ps.fonttype": 42,
            "path.simplify": False,      # keep exact paths (no weird merges)
            "pdf.compression": 0,        # avoid compression-related AI quirks
            "figure.dpi": 200,           # irrelevant for vector, fine to leave
        })

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
