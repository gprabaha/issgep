# src/socialgaze/features/psth_extractor.py

import os
import logging
from typing import Optional
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
    VALID_PSTH_TYPES = ["trial_wise", "face_obj", "int_non_int_face"]

    def __init__(self, config, gaze_data, spike_data, fixation_detector, interactivity_detector):
        self.config = config
        self.gaze_data = gaze_data
        self.spike_data = spike_data
        self.fixation_detector = fixation_detector
        self.interactivity_detector = interactivity_detector

        self.psth_per_trial: Optional[pd.DataFrame] = None
        self.avg_face_obj: Optional[pd.DataFrame] = None
        self.avg_int_non_int_face: Optional[pd.DataFrame] = None

    # ------------------------
    # == Public compute methods ==
    # ------------------------

    def compute_psth_per_trial(self):
        """Always compute and save PSTH per trial"""
        logger.info("Computing PSTH per trial...")

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
            with tqdm_joblib(tqdm(total=len(tasks), desc="PSTH extraction")):
                results = Parallel(n_jobs=self.config.num_cpus)(
                    delayed(_process_session_for_psth)(*args) for args in tasks
                )
        else:
            results = [_process_session_for_psth(*args) for args in tqdm(tasks)]

        rows = [r for result in results for r in result]
        self.psth_per_trial = pd.DataFrame(rows)
        save_df_to_pkl(self.psth_per_trial, self.config.psth_per_trial_path)
        logger.info(f"Saved PSTH per trial to {self.config.psth_per_trial_path}")


    def compute_avg_face_obj(self):
        """Always compute and save avg_face_obj"""
        if self.psth_per_trial is None:
            raise RuntimeError("PSTH per trial must be available. Fetch it first.")

        logger.info("Computing average PSTH: face vs object...")

        rows = []
        grouped = self.psth_per_trial.groupby(["unit_uuid", "category"])
        for (unit_uuid, category), group in grouped:
            if category not in ["face", "object"]:
                continue
            psth_array = np.stack(group["firing_rate"].apply(np.array))
            mean_psth = psth_array.mean(axis=0)
            row = group.iloc[0].to_dict()
            rows.append({
                "unit_uuid": unit_uuid,
                "category": category,
                "avg_firing_rate": mean_psth.tolist(),
                "region": row.get("region"),
                "channel": row.get("channel"),
                "session_name": row.get("session_name"),
                "run_number": row.get("run_number"),
                "agent": row.get("agent"),
            })

        self.avg_face_obj = pd.DataFrame(rows)
        save_df_to_pkl(self.avg_face_obj, self.config.avg_face_obj_path)
        logger.info(f"Saved avg_face_obj to {self.config.avg_face_obj_path}")


    def compute_avg_int_non_int_face(self):
        """Always compute and save avg_int_non_int_face"""
        if self.psth_per_trial is None:
            raise RuntimeError("PSTH per trial must be available. Fetch it first.")

        logger.info("Computing average PSTH: interactive vs non-interactive face...")

        df = self.psth_per_trial.copy()
        df = df[df["category"] == "face"]

        rows = []
        grouped = df.groupby(["unit_uuid", "is_interactive"])
        for (unit_uuid, is_interactive), group in grouped:
            psth_array = np.stack(group["firing_rate"].apply(np.array))
            mean_psth = psth_array.mean(axis=0)
            row = group.iloc[0].to_dict()
            rows.append({
                "unit_uuid": unit_uuid,
                "category": "face",
                "is_interactive": is_interactive,
                "avg_firing_rate": mean_psth.tolist(),
                "region": row.get("region"),
                "channel": row.get("channel"),
                "session_name": row.get("session_name"),
                "run_number": row.get("run_number"),
                "agent": row.get("agent"),
            })

        self.avg_int_non_int_face = pd.DataFrame(rows)
        save_df_to_pkl(self.avg_int_non_int_face, self.config.avg_int_non_int_face_path)
        logger.info(f"Saved avg_int_non_int_face to {self.config.avg_int_non_int_face_path}")


    # ------------------------
    # == Unified Fetch ==
    # ------------------------

    def fetch_psth(self, which: str) -> pd.DataFrame:
        if which not in self.VALID_PSTH_TYPES:
            raise ValueError(f"Invalid PSTH type '{which}'. Valid options: {self.VALID_PSTH_TYPES}")

        if which == "trial_wise":
            path = self.config.psth_per_trial_path
            if os.path.exists(path):
                self.psth_per_trial = load_df_from_pkl(path)
                logger.info(f"Loaded PSTH per trial from {path}")
            else:
                self.compute_psth_per_trial()
            return self.psth_per_trial

        elif which == "face_obj":
            path = self.config.avg_face_obj_path
            if os.path.exists(path):
                self.avg_face_obj = load_df_from_pkl(path)
                logger.info(f"Loaded avg_face_obj from {path}")
            else:
                self.compute_avg_face_obj()
            return self.avg_face_obj

        elif which == "int_non_int_face":
            path = self.config.avg_int_non_int_face_path
            if os.path.exists(path):
                self.avg_int_non_int_face = load_df_from_pkl(path)
                logger.info(f"Loaded avg_int_non_int_face from {path}")
            else:
                self.compute_avg_int_non_int_face()
            return self.avg_int_non_int_face


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
    do_smoothing,
    smoothing_sigma_bins,
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
