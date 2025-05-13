# src/socialgaze/features/psth_extractor.py

import os
import logging
from typing import Optional, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d

from socialgaze.utils.saving_utils import save_df_to_pkl
from socialgaze.utils.loading_utils import load_df_from_pkl

logger = logging.getLogger(__name__)


class PSTHExtractor:
    def __init__(self, config, gaze_data, spike_data, fixation_detector, interactivity_detector):
        """
        Initializes the PSTH extractor with pre-instantiated data managers.

        Args:
            config: NeuralConfig object with PSTH parameters and output paths.
            gaze_data: GazeData object with neural timeline and gaze info.
            spike_data: SpikeData object to load spike times.
            fixation_detector: FixationDetector object with loaded fixation events.
            interactivity_detector: InteractivityDetector with mutual density and interactivity periods.
        """
        self.config = config
        self.gaze_data = gaze_data
        self.spike_data = spike_data
        self.fixation_detector = fixation_detector
        self.interactivity_detector = interactivity_detector
        
        self.psth_mapping = {
            "trial_wise": "psth_per_trial",
            "by_category": "avg_psth_per_category",
            "by_interactivity": "avg_psth_per_category_and_interactivity",
        }

        self.psth_per_trial: Optional[pd.DataFrame] = None
        self.avg_psth_per_category: Optional[pd.DataFrame] = None
        self.avg_psth_per_category_and_interactivity: Optional[pd.DataFrame] = None

    # -------------------------------------------
    # == Methods to import or export PSTH data ==
    # -------------------------------------------

    def save_dataframes(self, which: Optional[List[str]] = None):
        """
        Saves the specified dataframe(s) to the configured output paths.

        Args:
            which (List[str], optional): List of {'trial_wise', 'by_category', 'by_interactivity'}.
                                        If None, saves all available.
        """
        if which is None:
            which = list(self.psth_mapping.keys())

        for key in which:
            if key not in self.psth_mapping:
                raise ValueError(f"Invalid save option '{key}'. Valid options: {list(self.psth_mapping.keys())}")

        os.makedirs(os.path.dirname(self.config.psth_per_trial_path), exist_ok=True)

        if "trial_wise" in which:
            if self.psth_per_trial is not None and not self.psth_per_trial.empty:
                save_df_to_pkl(self.psth_per_trial, self.config.psth_per_trial_path)
                logger.info(f"PSTH per trial data saved to {self.config.psth_per_trial_path}")
            else:
                logger.warning("No psth_per_trial data to save.")

        if "by_category" in which:
            if self.avg_psth_per_category is not None and not self.avg_psth_per_category.empty:
                save_df_to_pkl(self.avg_psth_per_category, self.config.avg_psth_per_category_path)
                logger.info(f"Avg PSTH per category data saved to {self.config.avg_psth_per_category_path}")
            else:
                logger.warning("No avg_psth_per_category data to save.")

        if "by_interactivity" in which:
            if self.avg_psth_per_category_and_interactivity is not None and not self.avg_psth_per_category_and_interactivity.empty:
                save_df_to_pkl(self.avg_psth_per_category_and_interactivity, self.config.avg_psth_per_category_and_interactivity_path)
                logger.info(f"Avg PSTH per category and interactivity data saved to {self.config.avg_psth_per_category_and_interactivity_path}")
            else:
                logger.warning("No avg_psth_per_category_and_interactivity data to save.")


    def load_dataframes(self, which: Optional[List[str]] = None):
        """
        Loads the specified dataframe(s) from the configured output paths.

        Args:
            which (List[str], optional): List of {'trial_wise', 'by_category', 'by_interactivity'}.
                                        If None, loads all available.
        """
        if which is None:
            which = list(self.psth_mapping.keys())

        for key in which:
            if key not in self.psth_mapping:
                raise ValueError(f"Invalid load option '{key}'. Valid options: {list(self.psth_mapping.keys())}")

        if "trial_wise" in which:
            if os.path.exists(self.config.psth_per_trial_path):
                self.psth_per_trial = load_df_from_pkl(self.config.psth_per_trial_path)
                logger.info(f"Loaded psth_per_trial from {self.config.psth_per_trial_path}")
            else:
                logger.warning(f"PSTH per trial file not found at {self.config.psth_per_trial_path}")

        if "by_category" in which:
            if os.path.exists(self.config.avg_psth_per_category_path):
                self.avg_psth_per_category = load_df_from_pkl(self.config.avg_psth_per_category_path)
                logger.info(f"Loaded avg_psth_per_category from {self.config.avg_psth_per_category_path}")
            else:
                logger.warning(f"Avg PSTH per category file not found at {self.config.avg_psth_per_category_path}")

        if "by_interactivity" in which:
            if os.path.exists(self.config.avg_psth_per_category_and_interactivity_path):
                self.avg_psth_per_category_and_interactivity = load_df_from_pkl(self.config.avg_psth_per_category_and_interactivity_path)
                logger.info(f"Loaded avg_psth_per_category_and_interactivity from {self.config.avg_psth_per_category_and_interactivity_path}")
            else:
                logger.warning(f"Avg PSTH per category and interactivity file not found at {self.config.avg_psth_per_category_and_interactivity_path}")


    def get_psth(
        self,
        which: str,
        session_name: Optional[str] = None,
        unit_uuid: Optional[str] = None,
        category: Optional[str] = None,
        is_interactive: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieves a PSTH dataframe based on type and optional filters.

        Args:
            which (str): One of {'trial_wise', 'by_category', 'by_interactivity'}.
            session_name (str, optional): Filter by session_name if available.
            unit_uuid (str, optional): Filter by unit_uuid if available.
            category (str, optional): Filter by category if available.
            is_interactive (str, optional): Filter by interactivity if available.

        Returns:
            pd.DataFrame: Filtered PSTH dataframe.
        """
        if which not in self.psth_mapping:
            raise ValueError(f"Invalid PSTH type '{which}'. Valid options are {list(self.psth_mapping.keys())}")

        attr_name = self.psth_mapping[which]
        df = getattr(self, attr_name)

        if df is None or df.empty:
            logger.info(f"DataFrame '{which}' is empty. Attempting to load...")
            self.load_dataframes(which=[which])
            df = getattr(self, attr_name)

        if df is None or df.empty:
            raise ValueError(f"Unable to load PSTH dataframe for '{which}'.")

        filters = {
            "session_name": session_name,
            "unit_uuid": unit_uuid,
            "category": category,
            "is_interactive": is_interactive,
        }

        # Apply filtering if possible
        for col, value in filters.items():
            if value is not None:
                if col not in df.columns:
                    raise ValueError(f"Cannot filter by '{col}' because it is not present in dataframe '{which}'.")
                df = df[df[col] == value]

        return df

    # ----------------------------------
    # == Methods to compute PSTH data ==
    # ----------------------------------

    def compute_psth_per_trial(self, overwrite: bool = False):
        """
        Computes PSTHs for each unit across sessions and runs using fixations from agent m1.
        Each fixation is tagged as interactive or not and associated with unit metadata.
        Results are stored in self.psth_per_trial.

        Args:
            overwrite (bool): If False (default), will load existing PSTH if available.
                              If True, will recompute PSTH even if file exists.
        """
        if not overwrite and os.path.exists(self.config.psth_per_trial_path):
            logger.info(f"PSTH per trial file exists at {self.config.psth_per_trial_path}. Loading instead of recomputing.")
            self.psth_per_trial = load_df_from_pkl(self.config.psth_per_trial_path)
            return

        # Otherwise recompute
        logger.info("Computing PSTHs from scratch...")

        if self.gaze_data.neural_timeline is None:
            self.gaze_data.load_dataframes(['neural_timeline'])
        if self.fixation_detector.fixations is None:
            self.fixation_detector.load_dataframes("fixations")
        if self.interactivity_detector.interactivity_periods is None:
            self.interactivity_detector.load_interactivity_periods()
        if self.spike_data.spike_df is None:
            self.spike_data.load_dataframes()

        interactivity_df = self.interactivity_detector.get_interactivity_periods()
        session_groups = self.fixation_detector.fixations.groupby("session_name")

        tasks = [
            (
                session_name,
                session_fix_df,
                self.spike_data.spike_df.query("session_name == @session_name"),
                interactivity_df,
                self.gaze_data,
                self.config
            )
            for session_name, session_fix_df in session_groups
        ]

        if self.config.use_parallel:
            logger.info(f"Running PSTH extraction in parallel across {len(tasks)} sessions.")
            with tqdm_joblib(tqdm(total=len(tasks), desc="PSTH extraction for session")):
                results = Parallel(n_jobs=self.config.num_cpus)(
                    delayed(_process_session_for_psth)(*args) for args in tasks
                )
        else:
            logger.info("Running PSTH extraction serially.")
            results = [_process_session_for_psth(*args) for args in tqdm(tasks, desc="PSTH extraction for session")]

        all_psth_rows = [row for session_result in results for row in session_result]
        self.psth_per_trial = pd.DataFrame(all_psth_rows)
        logger.info(f"Extracted {len(self.psth_per_trial)} PSTHs.")

        # Save after recomputing
        save_df_to_pkl(self.psth_per_trial, self.config.psth_per_trial_path)
        logger.info(f"Saved new PSTH per trial dataframe to {self.config.psth_per_trial_path}")


    def compute_avg_psth_per_category(self):
        """
        Computes average PSTH per unit across fixation categories,
        grouping only by unit_uuid (not session/run), but retaining session/run in the output.
        Stores result in self.avg_psth_per_category.
        """
        if self.psth_per_trial is None or self.psth_per_trial.empty:
            logger.error("PSTH per trial data is not available. Run compute_psth_per_trial first.")
            return

        logger.info("Computing average PSTH per unit per fixation category...")

        rows = []
        grouped = self.psth_per_trial.groupby(["unit_uuid", "category"])

        for (unit_uuid, category), group in grouped:
            psth_array = np.stack(group["firing_rate"].apply(np.array).values)
            mean_psth = psth_array.mean(axis=0)

            for _, row in group.iterrows():
                rows.append({
                    "session_name": row["session_name"],
                    "agent": row.get("agent"),
                    "unit_uuid": unit_uuid,
                    "region": row.get("region"),
                    "channel": row.get("channel"),
                    "category": category,
                    "avg_firing_rate": mean_psth.tolist()
                })
                break  # Only need one row to copy the metadata

        self.avg_psth_per_category = pd.DataFrame(rows)
        logger.info(f"Computed average PSTH for {len(self.avg_psth_per_category)} unit-category combinations.")


    def compute_avg_psth_per_category_and_interactivity(self):
        """
        Computes average PSTH per unit across fixation categories and interactivity,
        grouping only by unit_uuid, but retaining session/run in the output.
        Stores result in self.avg_psth_per_category_and_interactivity.
        """
        if self.psth_per_trial is None or self.psth_per_trial.empty:
            logger.error("PSTH per trial data is not available. Run compute_psth_per_trial first.")
            return

        logger.info("Computing average PSTH per unit per fixation category and interactivity...")

        rows = []
        grouped = self.psth_per_trial.groupby(["unit_uuid", "category", "is_interactive"])

        for (unit_uuid, category, is_interactive), group in grouped:
            psth_array = np.stack(group["firing_rate"].apply(np.array).values)
            mean_psth = psth_array.mean(axis=0)

            for _, row in group.iterrows():
                rows.append({
                    "session_name": row["session_name"],
                    "agent": row.get("agent"),
                    "unit_uuid": unit_uuid,
                    "region": row.get("region"),
                    "channel": row.get("channel"),
                    "category": category,
                    "is_interactive": is_interactive,
                    "avg_firing_rate": mean_psth.tolist()
                })
                break

        self.avg_psth_per_category_and_interactivity = pd.DataFrame(rows)
        logger.info(f"Computed average PSTH for {len(self.avg_psth_per_category_and_interactivity)} unit-category-interactivity combinations.")

#---------------------------------------------------
# == Support functions for PSTH computing methods ==
#---------------------------------------------------

def _process_session_for_psth(
    session_name,
    session_fix_df,
    spike_data_df,
    interactivity_df,
    gaze_data,
    config
):
    all_psth_rows = []
    m1_fix_df = session_fix_df[session_fix_df["agent"] == "m1"]

    for unit_row in spike_data_df.to_dict(orient="records"):
        spike_times = unit_row.get("spike_ts")
        unit_meta = {
            "region": unit_row.get("region"),
            "unit_uuid": unit_row.get("unit_uuid"),
            "channel": unit_row.get("channel")
        }

        for run_number, run_fix_df in m1_fix_df.groupby("run_number"):
            try:
                neural_times = gaze_data.neural_timeline.query(
                    "session_name == @session_name and run_number == @run_number"
                )["neural_timeline"].values[0]
            except IndexError:
                logger.warning(f"No neural timeline for session={session_name}, run={run_number}")
                continue

            inter_windows = list(zip(
                *interactivity_df.query(
                    "session_name == @session_name and run_number == @run_number"
                )[["start", "stop"]].values.T
            )) if not interactivity_df.empty else []

            psth_rows = _compute_psth_for_fixations_in_a_run(
                run_fix_df,
                neural_times,
                spike_times,
                inter_windows,
                unit_meta,
                config
            )
            all_psth_rows.extend(psth_rows)

    return all_psth_rows


def _compute_psth_for_fixations_in_a_run(fix_df, neural_times, spike_times, interactive_periods, unit_metadata, config):
    bin_size = config.psth_bin_size
    start_offset, end_offset = config.psth_window
    num_bins = int((end_offset - start_offset) / bin_size)

    rows_iterable = [row for _, row in fix_df.iterrows()]
    kwargs = {
        "neural_times": neural_times,
        "spike_times": spike_times,
        "bin_size": bin_size,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "num_bins": num_bins,
        "interactive_periods": interactive_periods,
        "unit_metadata": unit_metadata,
        "do_smoothing": getattr(config, "do_smoothing", False),
        "smoothing_sigma_bins": getattr(config, "smoothing_bin_sigma", 1.0),
    }

    return [
        r for r in (
            _process_fixation_row_for_psth(row, **kwargs)
            for row in rows_iterable
        ) if r is not None
    ]


def _process_fixation_row_for_psth(
    row,
    neural_times,
    spike_times,
    bin_size,
    start_offset,
    end_offset,
    num_bins,
    interactive_periods,
    unit_metadata,
    do_smoothing=False,
    smoothing_sigma_bins=2.0,
):
    idx_start = int(row["start"])
    idx_stop = int(row["stop"])
    if idx_start >= len(neural_times):
        return None

    t0 = neural_times[idx_start]
    window_start, window_end = t0 + start_offset, t0 + end_offset
    bins = np.linspace(window_start, window_end, num_bins + 1)
    counts, _ = np.histogram(np.asarray(spike_times).ravel(), bins=bins)
    firing_rate = counts.astype(np.float32) / bin_size

    if do_smoothing:
        firing_rate = gaussian_filter1d(firing_rate, sigma=smoothing_sigma_bins)

    is_interactive = any(start <= idx_start < stop for start, stop in interactive_periods)

    return {
        "session_name": row["session_name"],
        "run_number": row["run_number"],
        "agent": row["agent"],
        "fixation_start_idx": idx_start,
        "fixation_stop_idx": idx_stop,
        "category": row.get("category"),
        "is_interactive": "interactive" if is_interactive else "non-interactive",
        "firing_rate": firing_rate.tolist(),
        **unit_metadata
    }
