# src/socialgaze/features/psth_extractor.py

import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from socialgaze.utils.saving_utils import save_df_to_pkl, load_df_from_pkl

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

        self.psth_per_trial = None  # Will be populated by extract_psth()



    def compute_psth_per_trial(self):
        """
        Computes PSTHs for each unit across sessions and runs using fixations from agent m1.
        Each fixation is tagged as interactive or not and associated with unit metadata.
        Results are stored in self.psth_per_trial.
        """
        # Load required data if not already loaded
        if self.gaze_data.neural_timeline is None:
            self.gaze_data.load_dataframes(['neural_timeline'])
        if self.fixation_detector.fixations is None:
            self.fixation_detector.load_dataframes("fixations")
        if self.interactivity_detector.interactivity_periods is None:
            self.interactivity_detector.load_interactivity_periods()
        if self.spike_data.spike_df is None:
            self.spike_data.load_dataframes()

        interactivity_df = self.interactivity_detector.get_interactivity_periods()
        all_psth_rows = []

        session_groups = self.fixation_detector.fixations.groupby("session_name")

        for session_name, session_fix_df in tqdm(session_groups, desc="Processing sessions"):
            session_spikes_df = self.spike_data.spike_df.query("session_name == @session_name")
            m1_fix_df = session_fix_df[session_fix_df["agent"] == "m1"]

            for unit_row in session_spikes_df.to_dict(orient="records"):
                spike_times = unit_row.get("spike_times")
                unit_meta = {
                    "region": unit_row.get("region"),
                    "unit_uuid": unit_row.get("unit_uuid"),
                    "channel": unit_row.get("channel")
                }

                for run_number, run_fix_df in m1_fix_df.groupby("run_number"):
                    try:
                        neural_times = self.gaze_data.neural_timeline.query(
                            "session_name == @session_name and run_number == @run_number and agent == 'm1'"
                        )["neural_timeline"].values[0]
                    except IndexError:
                        logger.warning(f"No neural timeline for session={session_name}, run={run_number}, agent=m1")
                        continue

                    inter_windows = list(zip(
                        *interactivity_df.query(
                            "session_name == @session_name and run_number == @run_number"
                        )[["start", "stop"]].values.T
                    )) if not interactivity_df.empty else []

                    psth_rows = self._compute_psth_for_fixations_in_a_run(
                        fix_df=run_fix_df,
                        neural_times=neural_times,
                        spike_times=spike_times,
                        interactive_periods=inter_windows,
                        unit_metadata=unit_meta
                    )
                    all_psth_rows.extend(psth_rows)

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




    def _compute_psth_for_fixations_in_a_run(self, fix_df, neural_times, spike_times, interactive_periods, unit_metadata):
        """
        Computes PSTHs aligned to fixation start indices using neural timeline and spike times.
        Uses joblib for parallel execution over fixations if enabled in config.

        Args:
            fix_df (pd.DataFrame): Fixation events (agent m1 only).
            neural_times (np.ndarray): Neural timeline (seconds).
            spike_times (np.ndarray): Spike timestamps (seconds).
            interactive_periods (List[Tuple[int, int]]): Index-based start/stop tuples.
            unit_metadata (Dict): Filtered metadata: region, unit_uuid, channel.

        Returns:
            List[Dict]: One dict per fixation with PSTH, fixation and unit metadata.
        """
        if self.config.use_parallel and not hasattr(self.config, "num_cpus"):
            raise AttributeError("Parallel processing is enabled but `num_cpus` is not set in config.")

        bin_size = self.config.psth_bin_size
        start_offset, end_offset = self.config.psth_window
        num_bins = int((end_offset - start_offset) / bin_size)

        rows_iterable = [row for _, row in fix_df.iterrows()]

        compute = delayed(_process_fixation_row_for_psth)
        kwargs = {
            "neural_times": neural_times,
            "spike_times": spike_times,
            "bin_size": bin_size,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "num_bins": num_bins,
            "interactive_periods": interactive_periods,
            "unit_metadata": unit_metadata,
        }

        if self.config.use_parallel:
            results = Parallel(n_jobs=self.config.num_cpus)(
                compute(row, **kwargs) for row in rows_iterable
            )
        else:
            results = [compute(row, **kwargs) for row in rows_iterable]

        return [r for r in results if r is not None]


    def _process_fixation_row_for_psth(row, neural_times, spike_times, bin_size, start_offset, end_offset, num_bins, interactive_periods, unit_metadata):
        idx = int(row["start"])
        if idx >= len(neural_times):
            return None

        t0 = neural_times[idx]
        window_start, window_end = t0 + start_offset, t0 + end_offset
        bins = np.linspace(window_start, window_end, num_bins + 1)
        counts, _ = np.histogram(np.asarray(spike_times).ravel(), bins=bins)

        is_interactive = any(start <= idx < stop for start, stop in interactive_periods)

        return {
            "session_name": row["session_name"],
            "run_number": row["run_number"],
            "agent": row["agent"],
            "fixation_start_idx": idx,
            "fixation_category": row.get("fixation_category", "unknown"),
            "is_interactive": "interactive" if is_interactive else "non-interactive",
            "psth": counts.tolist(),
            **unit_metadata
        }



