# src/socialgaze/models/foraging_modeler.py

import logging
import pandas as pd
from typing import Optional, Literal, List

from socialgaze.config.foraging_config import ForagingConfig
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector

logger = logging.getLogger(__name__)


class ForagingModeler:
    def __init__(
        self,
        config: ForagingConfig,
        fixation_detector: FixationDetector,
        interactivity_detector: Optional[InteractivityDetector] = None,
    ):
        self.config = config
        self.fixation_detector = fixation_detector
        self.interactivity_detector = interactivity_detector

    def get_labeled_fixation_df(self) -> pd.DataFrame:
        """
        Loads and labels the fixation DataFrame with a 'period_type' column
        based on interactivity periods. This operates directly on the detector's df.
        """
        fixation_df = self.fixation_detector.fixations
        if fixation_df is None:
            fixation_df = self.fixation_detector.get_behavior_data("fixations")

        fixation_df = fixation_df.copy()

        if self.config.include_interactivity:
            assert self.interactivity_detector is not None
            inter_df = self.interactivity_detector.load_interactivity_periods()
            self.fixation_detector.gaze_data.load_dataframes("run_lengths")
            run_lengths = self.fixation_detector.gaze_data.run_lengths
            fixation_df = _label_fixation_df_with_interactivity(fixation_df, inter_df, run_lengths)
        else:
            fixation_df["period_type"] = "full"

        return fixation_df

    def get_sequences_by_agent(
        self,
        agent: str,
        period_type: Optional[str] = None,
        category_subset: Optional[List[str]] = None,
    ) -> dict:
        """
        Returns a dict keyed by (session_name, run_number) containing the sequence of fixations
        for a specific agent. Each entry is a DataFrame of fixations in order.
        """
        df = self.get_labeled_fixation_df()
        df = df[df["agent"] == agent]
        if period_type:
            df = df[df["period_type"] == period_type]
        if category_subset:
            df = df[df["category"].isin(category_subset)]

        grouped = df.sort_values("start").groupby(["session_name", "run_number"])
        return {key: group.reset_index(drop=True) for key, group in grouped}

    # === Modeling entrypoints ===

    def prepare_mvt_data(
        self,
        agent: str,
        period_type: Optional[str] = None,
        category_subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepares sequence-level data for Marginal Value Theorem analysis.
        Returns a flat DataFrame with patch duration, category, and transitions.
        """
        sequences = self.get_sequences_by_agent(agent, period_type, category_subset)

        all_rows = []
        for (session, run), df in sequences.items():
            prev_category = None
            for i, row in df.iterrows():
                duration = row["stop"] - row["start"] + 1
                row_data = {
                    "session_name": session,
                    "run_number": run,
                    "agent": agent,
                    "period_type": row["period_type"],
                    "category": row["category"],
                    "duration": duration,
                    "prev_category": prev_category,
                    "next_category": df.iloc[i + 1]["category"] if i + 1 < len(df) else None,
                }
                all_rows.append(row_data)
                prev_category = row["category"]

        return pd.DataFrame(all_rows)

    def fit_marginal_value_model(self, agent: str, period_type: str = "full"):
        """
        Fits a foraging model where decisions to leave a patch depend on category, duration, and history.
        (Currently a placeholder for logistic regression, RL, or semi-Markov model.)
        """
        mvt_df = self.prepare_mvt_data(agent, period_type)

        # Example: probability of leaving after duration D
        mvt_df["is_leave"] = mvt_df["category"] != mvt_df["next_category"]

        from sklearn.linear_model import LogisticRegression
        import numpy as np

        X = mvt_df[["duration"]].to_numpy()
        y = mvt_df["is_leave"].astype(int)

        model = LogisticRegression().fit(X, y)
        logger.info(f"[{agent} | {period_type}] MVT leave model coef={model.coef_[0][0]:.4f}")
        return model, mvt_df

    def fit_semi_markov_model(self, agent: str, period_type: str = "full"):
        """
        Placeholder for fitting a semi-Markov model over patch categories.
        """
        seqs = self.get_sequences_by_agent(agent, period_type)
        raise NotImplementedError("Semi-Markov model fitting not yet implemented.")


def _label_fixation_df_with_interactivity(fixation_df, inter_df, run_lengths_df):
    fixation_df = fixation_df.copy()
    fixation_df["period_type"] = "non_interactive"

    grouped = fixation_df.groupby(["session_name", "run_number"])
    for (session, run), run_df in grouped:
        try:
            full_stop = run_lengths_df.query(
                "session_name == @session and run_number == @run"
            )["stop"].values[0]
        except IndexError:
            continue

        interactive_periods = inter_df.query(
            "session_name == @session and run_number == @run"
        )
        intervals = list(zip(interactive_periods["start"], interactive_periods["stop"]))
        inverted = _invert_intervals(intervals, (0, full_stop))

        for idx, row in run_df.iterrows():
            s, e = row["start"], row["stop"]
            if any(start <= s and e <= stop for start, stop in intervals):
                fixation_df.at[idx, "period_type"] = "interactive"
            elif any(start <= s and e <= stop for start, stop in inverted):
                fixation_df.at[idx, "period_type"] = "non_interactive"

    return fixation_df


def _invert_intervals(intervals, full_range):
    full_start, full_stop = full_range
    intervals = sorted(intervals)
    inverted = []
    last = full_start
    for start, stop in intervals:
        if last < start:
            inverted.append((last, start - 1))
        last = max(last, stop + 1)
    if last <= full_stop:
        inverted.append((last, full_stop))
    return inverted
