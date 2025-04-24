# src/socialgaze/features/psth_extractor.py

import os
import logging
from typing import Optional, List, Tuple, Dict
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
        self.psth_per_trial: Optional[pd.DataFrame] = None

    def compute_psth_per_trial(self):
        """
        Computes PSTHs for each unit across sessions and runs using fixations from agent m1.
        Each fixation is tagged as interactive or not and associated with unit metadata.
        Results are stored in self.psth_per_trial.
        """
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

    def save_dataframes(self):
        """
        Saves the PSTH dataframe to the configured output path.
        """
        if self.psth_per_trial is None or self.psth_per_trial.empty:
            logger.warning("No PSTH data to save.")
            return
        save_df_to_pkl(self.psth_per_trial, self.config.psth_per_trial_path)
        logger.info(f"PSTH data saved to {self.config.psth_per_trial_path}")

    def load_dataframes(self):
        """
        Loads PSTH data from the configured output path into self.psth_df.
        """
        if not os.path.exists(self.config.psth_per_trial_path):
            logger.error(f"PSTH file not found at: {self.config.psth_per_trial_path}")
            return
        self.psth_per_trial = load_df_from_pkl(self.config.psth_per_trial_path)
        logger.info(f"Loaded PSTH data from {self.config.psth_per_trial_path}")


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
    idx = int(row["start"])
    if idx >= len(neural_times):
        return None

    t0 = neural_times[idx]
    window_start, window_end = t0 + start_offset, t0 + end_offset
    bins = np.linspace(window_start, window_end, num_bins + 1)
    counts, _ = np.histogram(np.asarray(spike_times).ravel(), bins=bins)
    firing_rate = counts.astype(np.float32) / bin_size

    if do_smoothing:
        firing_rate = gaussian_filter1d(firing_rate, sigma=smoothing_sigma_bins)

    is_interactive = any(start <= idx < stop for start, stop in interactive_periods)

    return {
        "session_name": row["session_name"],
        "run_number": row["run_number"],
        "agent": row["agent"],
        "fixation_start_idx": idx,
        "fixation_category": row.get("fixation_category", "unknown"),
        "is_interactive": "interactive" if is_interactive else "non-interactive",
        "firing_rate": firing_rate.tolist(),
        **unit_metadata
    }
