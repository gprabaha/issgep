# src/socialgaze/features/crosscorr_calculator.py

import pdb
import logging
import os
from typing import Optional, Dict, List, Tuple
from glob import glob
from pathlib import Path

import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import fftconvolve
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp
from collections import defaultdict

from socialgaze.config.crosscorr_config import CrossCorrConfig
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.utils.loading_utils import load_df_from_pkl
from socialgaze.utils.saving_utils import save_df_to_pkl
from socialgaze.utils.hpc_utils import (
    generate_crosscorr_job_file,
    submit_dsq_array_job,
    track_job_completion
)
from socialgaze.utils.path_utils import CrossCorrPaths

logger = logging.getLogger(__name__)


class CrossCorrCalculator:
    """
    Computes cross-correlations and shuffled cross-correlations between binary behavioral vectors 
    from two agents across experimental sessions and runs.

    Attributes:
        config (CrossCorrConfig): Configuration object containing parameters and paths.
        fixation_detector (FixationDetector): Provides access to binary fixation vectors.
        interactivity_detector (Optional[InteractivityDetector]): Provides interactivity period data.
    """

    def __init__(
        self,
        config: CrossCorrConfig,
        fixation_detector: FixationDetector,
        interactivity_detector: Optional[InteractivityDetector] = None,
    ):
        self.config = config
        self.fixation_detector = fixation_detector
        self.interactivity_detector = interactivity_detector
        self.paths = CrossCorrPaths(config)


    def compute_crosscorrelations(self, by_interactivity_period: bool = False):
        """
        Computes cross-correlations for each session/run and agent-behavior pair.

        Args:
            by_interactivity_period (bool): If True, computes separate cross-correlations for 
                                            interactive and non-interactive periods using the
                                            InteractivityDetector. Otherwise computes over full duration.
        """

        if by_interactivity_period:
            assert self.interactivity_detector is not None, "InteractivityDetector must be provided."
            logger.info("Computing cross-correlations by interactivity period...")
            inter_df = self.interactivity_detector.load_interactivity_periods()
        else:
            logger.info("Computing full-period cross-correlations...")

        for a1, b1, a2, b2 in tqdm(self.config.crosscorr_agent_behavior_pairs, desc="Agent-behavior pairs"):
            try:
                df1 = self.fixation_detector.get_binary_vector_df(b1)
                if b1==b2: # if we are comparing teh same behavior between two agents
                    df2 = df1
                else:
                    df2 = self.fixation_detector.get_binary_vector_df(b2)
            except FileNotFoundError:
                logger.warning(f"Missing binary vector: {b1} or {b2}")
                continue

            df1 = df1[df1["agent"] == a1]
            df2 = df2[df2["agent"] == a2]
            if df1.empty or df2.empty:
                continue

            merged = pd.concat([df1, df2], ignore_index=True)
            grouped = merged.groupby(["session_name", "run_number"])

            period_types = ["interactive", "non_interactive"] if by_interactivity_period else ["full"]

            for period_type in period_types:
                group_list = list(grouped)
                if self.config.show_inner_tqdm:
                    group_iter = tqdm(group_list, desc=f"{a1}_{b1} vs {a2}_{b2} [{period_type}]", leave=False)
                else:
                    group_iter = group_list

                if self.config.use_parallel:
                    results = Parallel(n_jobs=self.config.num_cpus)(
                        delayed(_process_one_session_run_crosscorr)(
                            session, run, run_df, a1, b1, a2, b2, period_type,
                            inter_df if by_interactivity_period else None,
                            self.config
                        )
                        for (session, run), run_df in group_iter
                    )
                else:
                    results = [
                        _process_one_session_run_crosscorr(
                            session, run, run_df, a1, b1, a2, b2, period_type,
                            inter_df if by_interactivity_period else None,
                            self.config
                        )
                        for (session, run), run_df in group_iter
                    ]

                all_rows = [df for df in results if isinstance(df, pd.DataFrame)]

                if all_rows:
                    full_df = pd.concat(all_rows, ignore_index=True)
                    out_path = self.paths.get_obs_crosscorr_path(a1, b1, a2, b2, period_type)
                    save_df_to_pkl(full_df, out_path)
                    logger.info(f"Saved observed crosscorr df to: {out_path}"
                        "\nHead of calculated dataframe:"
                        f"\n{full_df.head()}"
                    )
        logger.info("Cross-correlation computation complete.")


    def compute_shuffled_crosscorrelations(self, by_interactivity_period: bool = False):
        """
        Prepares and submits a SLURM job array to compute shuffled cross-correlations
        for all session/run and agent-behavior combinations.
        If config.run_single_test_case is True, only one randomly selected task is executed locally.
        Args:
            by_interactivity_period (bool): Whether to compute per-period (interactive vs. non-interactive).
        """

        logger.info(f"Preparing shuffled cross-correlation job array "
                    f"({'by interactivity' if by_interactivity_period else 'full'})...")

        tasks = []
        period_types = ["interactive", "non_interactive"] if by_interactivity_period else ["full"]

        for a1, b1, a2, b2 in self.config.crosscorr_agent_behavior_pairs:
            for session in self.config.session_names:
                runs = self.config.runs_by_session.get(session, [])
                for run in runs:
                    for period_type in period_types:
                        tasks.append((session, run, a1, b1, a2, b2, period_type))

        if not tasks:
            logger.warning("No valid tasks to run.")
            return

        # Run one random test case if test flag is set
        if self.config.run_single_test_case:
            test_task = random.choice(tasks)
            logger.info(f"Running single test task: {test_task}")
            self.compute_shuffled_crosscorrelations_for_single_run(*test_task)
            return

        generate_crosscorr_job_file(tasks, self.config)
        job_id = submit_dsq_array_job(self.config)
        track_job_completion(job_id)
        self.combine_and_save_shuffled_results()


    def compute_shuffled_crosscorrelations_for_single_run(self, session, run, a1, b1, a2, b2, period_type="full"):
        session = str(session).strip()
        run = str(run).strip()
        a1, b1 = a1.lower().strip(), b1.strip()
        a2, b2 = a2.lower().strip(), b2.strip()

        logger.info(f"Computing shuffled crosscorr: session={session}, run={run}, "
                    f"a1={a1}, b1={b1}, a2={a2}, b2={b2}, period_type={period_type}")

        # --- Load vectors ---
        v1, v2 = self._load_and_prepare_vectors_for_run(session, run, a1, b1, a2, b2)
        if v1 is None or v2 is None:
            return

        # --- Get periods ---
        periods = self._get_valid_periods_for_run(session, run, period_type, len(v1))

        if periods is None or len(periods) == 0:
            logger.warning("No valid periods")
            return

        # --- Prepare shuffle jobs ---
        shuffle_args = [(v1, v2, periods, self.config) for _ in range(self.config.num_shuffles)]

        # --- Compute shuffled correlations ---
        logger.info(f"Running {self.config.num_shuffles} shuffles "
                    f"{'in parallel' if self.config.use_parallel else 'serially'}")

        if self.config.use_parallel:
            corrs = list(tqdm(
                Parallel(n_jobs=self.config.num_cpus)(
                    delayed(_compute_one_shuffled_crosscorr_for_run)(*args)
                    for args in shuffle_args
                ),
                total=self.config.num_shuffles,
                desc="Shuffled crosscorr (parallel)"
            ))
        else:
            corrs = [
                _compute_one_shuffled_crosscorr_for_run(*args)
                for args in tqdm(shuffle_args, desc="Shuffled crosscorr (serial)")
            ]

        if not corrs:
            logger.warning(f"No cross-correlation results for session={session}, run={run}")
            return

        # --- Extract lags and summarize ---
        lags, _ = _compute_normalized_crosscorr(
            np.zeros_like(v1), np.zeros_like(v2),
            max_lag=self.config.max_lag,
            normalize=self.config.normalize,
            use_energy_norm=self.config.use_energy_norm,
        )

        corr_values = np.stack(corrs)
        mean_corr = np.mean(corr_values, axis=0)
        std_corr = np.std(corr_values, axis=0)

        out_df = pd.DataFrame({
            "session_name": session,
            "run_number": run,
            "agent1": a1,
            "agent2": a2,
            "behavior1": b1,
            "behavior2": b2,
            "lag": [lags],
            "crosscorr_mean": [mean_corr],
            "crosscorr_std": [std_corr],
            "period_type": period_type
        })

        logger.info(f"Finished session={session}, run={run}, period_type={period_type}\n{out_df.head()}")

        # --- Save output ---
        out_path = self.paths.get_shuffled_temp_path(session, run, a1, b1, a2, b2, period_type)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_df_to_pkl(out_df, out_path)
        logger.info(f"Saved TEMP shuffled result: {out_path}")


    def _load_and_prepare_vectors_for_run(self, session, run, a1, b1, a2, b2):
        try:
            df1 = self.fixation_detector.get_binary_vector_df(b1)
            df2 = df1 if b1 == b2 else self.fixation_detector.get_binary_vector_df(b2)
        except FileNotFoundError:
            logger.warning(f"Missing binary vector: {b1} or {b2}")
            return None, None

        for df in [df1, df2]:
            df["agent"] = df["agent"].str.lower().str.strip()
            df["session_name"] = df["session_name"].str.strip()
            df["run_number"] = df["run_number"].astype(str).str.strip()

        df1 = df1[(df1["agent"] == a1) & (df1["session_name"] == session) & (df1["run_number"] == run)]
        df2 = df2[(df2["agent"] == a2) & (df2["session_name"] == session) & (df2["run_number"] == run)]

        if df1.empty or df2.empty:
            logger.warning(f"No data for session {session}, run {run}")
            return None, None

        run_df = pd.concat([df1, df2], ignore_index=True)
        return _get_vectors_for_run(run_df, a1, b1, a2, b2)


    def _get_valid_periods_for_run(self, session, run, period_type, vector_length):
        if period_type == "full":
            return [(0, vector_length - 1)]

        inter_df = self.interactivity_detector.load_interactivity_periods()
        periods = _get_periods_for_run(inter_df, session, run, period_type, vector_length)
        
        if periods is None or len(periods) == 0:
            logger.warning(f"No valid periods for session={session}, run={run}, period_type={period_type}")
        return periods


    def combine_and_save_shuffled_results(self):
        """
        Combines run-level shuffled crosscorr files into one file per (a1, b1, a2, b2, period_type)
        using grouped temp paths from CrossCorrPaths. Deletes temp files after combining.
        """

        grouped_paths = self.paths.get_grouped_shuffled_temp_paths()

        for (a1, b1, a2, b2, period_type), file_list in grouped_paths.items():
            dfs = [pd.read_pickle(f) for f in file_list]
            combined = pd.concat(dfs, ignore_index=True)

            pd.set_option("display.width", 0)
            pd.set_option("display.max_columns", None)
            logger.info(f"Resultant dataframe for {a1}_{b1} vs {a2}_{b2} [{period_type}]:\n{combined.head()}")

            out_path = self.paths.get_shuffled_final_path(a1, b1, a2, b2, period_type)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_pickle(out_path)
            logger.info(f"Saved COMBINED shuffled cross-correlation to: {out_path}")

            for f in file_list:
                os.remove(f)
                logger.debug(f"Deleted temp file: {f}")



    def analyze_crosscorr_vs_shuffled_per_pair(self):
        """
        Compares observed vs shuffled crosscorrelations for each behavior pair and period,
        grouped by monkey identity pairs from self.config.ephys_days_and_monkeys.
        Handles lag truncation and saves full results.
        """
        results = []
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 0)

        session_to_monkey_pair = self.config.ephys_days_and_monkeys_df.set_index("session_name")[["m1", "m2"]].to_dict("index")

        for a1, b1, a2, b2 in self.config.crosscorr_agent_behavior_pairs:
            for period_type in ["full", "interactive", "non_interactive"]:
                obs_path = self.paths.get_obs_crosscorr_path(a1, b1, a2, b2, period_type)
                shuffled_path = self.paths.get_shuffled_final_path(a1, b1, a2, b2, period_type)

                if not obs_path.exists() or not shuffled_path.exists():
                    logger.warning(f"Missing observed or shuffled file for {a1}-{b1} vs {a2}-{b2} [{period_type}]")
                    continue

                try:
                    observed_df = pd.read_pickle(obs_path)
                    shuffled_df = pd.read_pickle(shuffled_path)
                except Exception as e:
                    logger.warning(f"Error loading data for {a1}-{b1} vs {a2}-{b2} [{period_type}]: {e}")
                    continue

                obs_groups = observed_df.groupby(["session_name", "run_number"])
                shuffle_groups = shuffled_df.groupby(["session_name", "run_number"])

                delta_dict = defaultdict(list)
                lag_dict = {}

                for (session, run), obs_group in obs_groups:
                    session = str(session)
                    run = str(run)

                    if session not in session_to_monkey_pair:
                        logger.warning(f"No monkey ID info for session {session}")
                        continue
                    monkey_pair = tuple(session_to_monkey_pair[session].values())

                    try:
                        shuffle_group = shuffle_groups.get_group((session, run))
                    except KeyError:
                        logger.warning(f"No shuffled data for session {session} run {run}")
                        continue

                    try:
                        lags = obs_group.iloc[0]["lag"]
                        obs_corr = obs_group.iloc[0]["crosscorr"]

                        # Shuffled vector is stored as a row with single array
                        shuffled_mean = shuffle_group.iloc[0]["crosscorr_mean"]

                        # Ensure symmetry
                        min_len = min(len(obs_corr), len(shuffled_mean))
                        if min_len % 2 == 0:
                            min_len -= 1
                        center_obs = len(obs_corr) // 2
                        center_shuf = len(shuffled_mean) // 2

                        obs_corr = obs_corr[center_obs - min_len // 2: center_obs + min_len // 2 + 1]
                        shuffled_mean = shuffled_mean[center_shuf - min_len // 2: center_shuf + min_len // 2 + 1]
                        lags = lags[center_obs - min_len // 2: center_obs + min_len // 2 + 1]

                        delta = obs_corr - shuffled_mean
                        delta_dict[monkey_pair].append(delta)
                        lag_dict[monkey_pair] = lags

                    except Exception as e:
                        logger.warning(f"Error computing delta for session {session} run {run}: {e}")
                        continue

                for monkey_pair, deltas in delta_dict.items():
                    if not deltas:
                        continue
                    min_len = min(len(d) for d in deltas)
                    if min_len < 2:
                        logger.warning(f"Too short vectors for {monkey_pair}, skipping")
                        continue

                    truncated_deltas = np.stack([d[:min_len] for d in deltas])
                    lags = lag_dict[monkey_pair][:min_len]

                    mean_delta = truncated_deltas.mean(axis=0)
                    t_stat, p_vals = ttest_1samp(truncated_deltas, popmean=0, axis=0, nan_policy="omit")

                    results.append({
                        "comparison": self.paths.get_comparison_name(a1, b1, a2, b2),
                        "period_type": period_type,
                        "monkey_pair": monkey_pair,
                        "lags": lags,
                        "mean_delta": mean_delta,
                        "t_stat": t_stat,
                        "p_values": p_vals,
                        "n_runs": len(deltas)
                    })

        self.save_crosscorr_analysis_results(results)
        return results


    def save_crosscorr_analysis_results(self, results):
        """
        Saves results to: config.output_dir/results/mean_minus_shuffled_crosscorr_results.pkl
        """
        out_path = self.paths.get_analysis_output_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(results, out_path)
        print(f"Saved delta crosscorr results to {out_path}")



# ------------------------
# Cross-correlation helpers
# ------------------------


def _process_one_session_run_crosscorr(session, run, run_df, a1, b1, a2, b2, period_type, inter_df, config):
    v1, v2 = _get_vectors_for_run(run_df, a1, b1, a2, b2)
    if v1 is None or v2 is None:
        return []

    if period_type == "full":
        periods = [(0, len(v1) - 1)]
    else:
        periods = _get_periods_for_run(inter_df, session, run, period_type, len(v1))
        if periods is None or len(periods) == 0:
            return []

    return _compute_crosscorr_for_periods(session, run, a1, a2, b1, b2, periods, v1, v2, period_type, config)


def _get_vectors_for_run(run_df, a1, b1, a2, b2):
    agent_data = {
        (row["agent"], row["behavior_type"]): row["binary_vector"]
        for _, row in run_df.iterrows()
    }
    if (a1, b1) not in agent_data or (a2, b2) not in agent_data:
        logger.warning(f"Missing binary vector for {a1}-{b1} or {a2}-{b2} in run_df.")
        return None, None
    return agent_data[(a1, b1)], agent_data[(a2, b2)]


def _get_periods_for_run(inter_df, session, run, period_type, full_len):
    run_periods = inter_df[
        (inter_df.session_name == session) &
        (inter_df.run_number == run)
    ]
    if run_periods.empty:
        logger.warning(f"Binary vector data was empty for session {session}, run {run}, and period_type {period_type}")
        return []
    inter = run_periods[["start", "stop"]].values
    return inter if period_type == "interactive" else _compute_complement_periods(inter, [(0, full_len - 1)])


def _compute_complement_periods(included: np.ndarray, total: List[tuple]) -> List[tuple]:
    if included.size == 0:
        return total
    complement = []
    current = total[0][0]
    for start, stop in included:
        if current < start:
            complement.append((current, start - 1))
        current = stop + 1
    if current <= total[0][1]:
        complement.append((current, total[0][1]))
    return complement


def _compute_crosscorr_for_periods(session, run, a1, a2, b1, b2, periods, v1, v2, period_type, config):
    """
    Computes cross-correlation across the union of all valid periods by filling zero-initialized
    vectors and computing a single cross-correlation on the result.

    Args:
        session (str): Session name.
        run (str): Run number.
        a1, a2 (str): Agent names.
        b1, b2 (str): Behavior types.
        periods (list of tuples): List of (start, stop) indices.
        v1, v2 (np.ndarray): Binary vectors for agent1 and agent2.
        period_type (str): Period type ('full', 'interactive', etc.).
        config: Configuration object containing parameters like max_lag.

    Returns:
        pd.DataFrame: Single-row dataframe with cross-correlation result.
    """
    full_seg1 = np.zeros_like(v1, dtype=np.float32)
    full_seg2 = np.zeros_like(v2, dtype=np.float32)

    for start, stop in periods:
        if start >= len(v1) or stop >= len(v1) or start >= len(v2) or stop >= len(v2):
            logger.warning(f"Skipping invalid period ({start}, {stop}) for session {session}, run {run}, len v1: {len(v1)}, len v2: {len(v2)}")
            continue
        full_seg1[start:stop + 1] = v1[start:stop + 1]
        full_seg2[start:stop + 1] = v2[start:stop + 1]

    lags, corr = _compute_normalized_crosscorr(
        full_seg1,
        full_seg2,
        max_lag=config.max_lag,
        normalize=config.normalize,
        use_energy_norm=config.use_energy_norm
    )

    df = pd.DataFrame({
        "session_name": session,
        "run_number": run,
        "agent1": a1,
        "agent2": a2,
        "behavior1": b1,
        "behavior2": b2,
        "lags": [lags],
        "crosscorr": [corr],
        "period_type": period_type
    })

    return df



def _compute_normalized_crosscorr(x: np.ndarray, y: np.ndarray, max_lag: Optional[int] = None, normalize: bool = True, use_energy_norm: bool = False):
    x = x.astype(float)
    y = y.astype(float)
    corr_full = fftconvolve(x, y[::-1], mode="full")
    lags_full = np.arange(-len(y) + 1, len(x))
    corr = corr_full
    lags = lags_full

    if max_lag is not None:
        center = len(corr_full) // 2
        corr = corr_full[center - max_lag:center + max_lag + 1]
        lags = lags_full[center - max_lag:center + max_lag + 1]

    if normalize:
        if use_energy_norm:
            norm = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
            if norm > 0:
                corr /= norm
            else:
                corr[:] = 0
        else:
            x_std = np.std(x)
            y_std = np.std(y)
            if x_std > 0 and y_std > 0:
                corr /= (len(x) * x_std * y_std)
            else:
                corr[:] = 0

    return lags, corr



# ------------------------
# Shuffling-related helpers
# ------------------------


def _compute_one_shuffled_crosscorr_for_run(
    v1: np.ndarray,
    v2: np.ndarray,
    periods: List[Tuple[int, int]],
    config
) -> np.ndarray:
    """
    Generates one pair of full-length shuffled vectors and computes cross-correlation.
    """
    full_vec1 = np.zeros_like(v1, dtype=int)
    full_vec2 = np.zeros_like(v2, dtype=int)

    for start, stop in periods:
        seg1 = v1[start:stop + 1]
        seg2 = v2[start:stop + 1]
        if len(seg1) < 2 or len(seg2) < 2:
            continue

        try:
            shuffled_seg1 = _generate_one_shuffled_vector_for_segment(
                seg1, len(seg1), config.make_shuffle_stringent
            )
            shuffled_seg2 = _generate_one_shuffled_vector_for_segment(
                seg2, len(seg2), config.make_shuffle_stringent
            )
            full_vec1[start:stop + 1] = shuffled_seg1
            full_vec2[start:stop + 1] = shuffled_seg2
        except Exception as e:
            logger.warning(f"Shuffling failed at period {start}-{stop}: {e}")
            continue

    _, corr = _compute_normalized_crosscorr(
        full_vec1,
        full_vec2,
        max_lag=config.max_lag,
        normalize=config.normalize,
        use_energy_norm=config.use_energy_norm,
    )
    return corr


def _generate_one_shuffled_vector_for_segment(
    original_segment: np.ndarray,
    segment_length: int,
    stringent: bool
) -> np.ndarray:
    """
    Generate a single shuffled version of a binary segment.
    """
    ones = np.where(original_segment == 1)[0]
    if len(ones) == 0:
        return np.zeros(segment_length, dtype=int)

    starts = ones[np.diff(np.concatenate(([-2], ones))) > 1]
    stops = ones[np.diff(np.concatenate((ones, [segment_length + 2]))) > 1]
    fixation_durations = [stop - start + 1 for start, stop in zip(starts, stops)]
    total_fix_dur = sum(fixation_durations)
    non_fix_dur = segment_length - total_fix_dur

    return _generate_single_shuffled_vector(fixation_durations, non_fix_dur, segment_length, stringent)


def _generate_shuffled_vectors_for_run_in_parallel(
    original_vector: np.ndarray,
    run_length: int,
    num_shuffles: int,
    num_cpus: int,
    stringent: bool
) -> List[np.ndarray]:
    """
    Generates shuffled versions of a binary vector by preserving fixation durations 
    and redistributing them within the run length.

    Args:
        original_vector (np.ndarray): Binary vector (1 for fixation, 0 for no fixation).
        run_length (int): Total length of the run.
        num_shuffles (int): Number of shuffled vectors to generate.
        num_cpus (int): Number of CPUs to use for parallel generation.
        stringent (bool): If True, preserves inter-fixation intervals more strictly.

    Returns:
        List[np.ndarray]: List of shuffled binary vectors.
    """
    ones = np.where(original_vector == 1)[0]
    if len(ones) == 0:
        return [np.zeros(run_length, dtype=int) for _ in range(num_shuffles)]

    # Identify contiguous 1-segments as fixation intervals
    starts = ones[np.diff(np.concatenate(([-2], ones))) > 1]
    stops = ones[np.diff(np.concatenate((ones, [run_length + 2]))) > 1]
    fixation_durations = [stop - start + 1 for start, stop in zip(starts, stops)]
    total_fix_dur = sum(fixation_durations)
    non_fix_dur = run_length - total_fix_dur

    return Parallel(n_jobs=num_cpus)(
        delayed(_generate_single_shuffled_vector)(
            fixation_durations, non_fix_dur, run_length, stringent
        ) for _ in range(num_shuffles)
    )


def _generate_single_shuffled_vector(
    fixation_durations: List[int],
    non_fixation_total: int,
    run_length: int,
    stringent: bool
) -> np.ndarray:
    if not fixation_durations:
        return np.zeros(run_length, dtype=int)

    if stringent:
        non_fixation_durations = _generate_uniform_partitions(non_fixation_total, len(fixation_durations) + 1)
        segments = _interleave_segments(fixation_durations, non_fixation_durations)
    else:
        return np.random.permutation(np.array([1 if i in fixation_durations else 0 for i in range(run_length)]))

    return _construct_shuffled_vector(segments, run_length)


def _generate_uniform_partitions(total: int, n_parts: int) -> List[int]:
    if n_parts == 1:
        return [total]
    cuts = np.sort(np.random.choice(range(1, total), n_parts - 1, replace=False))
    parts = np.diff(np.concatenate([[0], cuts, [total]]))
    while sum(parts) != total:
        diff = total - sum(parts)
        idx = np.random.choice(len(parts), size=abs(diff), replace=True)
        for i in idx:
            if diff > 0:
                parts[i] += 1
            elif parts[i] > 1:
                parts[i] -= 1
    return parts.tolist()


def _interleave_segments(fix_durs: List[int], non_fix_durs: List[int]) -> List[Tuple[int, int]]:
    np.random.shuffle(fix_durs)
    np.random.shuffle(non_fix_durs)
    segments = []
    for i in range(len(fix_durs)):
        segments.append((non_fix_durs[i], 0))
        segments.append((fix_durs[i], 1))
    segments.append((non_fix_durs[-1], 0))
    return segments


def _construct_shuffled_vector(segments: List[Tuple[int, int]], run_length: int) -> np.ndarray:
    vec = np.zeros(run_length, dtype=int)
    idx = 0
    for dur, val in segments:
        if idx >= run_length:
            break
        end = min(idx + dur, run_length)
        if val == 1:
            vec[idx:end] = 1
        idx += dur
    return vec








import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from datetime import datetime
import numpy as np

def plot_crosscorr_deltas_combined(results, config, alpha=0.05):
    """
    For each monkey pair:
        - One figure.
        - Rows = period_type.
        - Each subplot overlays all comparisons with unique color.
    Saves to: config.output_dir/plots/mean_minus_shuffled_crosscorr/<date>/
    """
    grouped = defaultdict(list)
    for res in results:
        pair_key = f"{res['monkey_pair'][0]}-{res['monkey_pair'][1]}"
        grouped[pair_key].append(res)

    plot_dir = config.output_dir / "plots" / "mean_minus_shuffled_crosscorr" / datetime.now().strftime("%Y-%m-%d")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for monkey_pair, res_list in grouped.items():
        comparisons = sorted(set(r["comparison"] for r in res_list))
        periods = sorted(set(r["period_type"] for r in res_list))

        # Assign a unique color to each comparison
        cmap = cm.get_cmap("tab10", len(comparisons))
        comp_color_map = {comp: cmap(i) for i, comp in enumerate(comparisons)}

        fig, axes = plt.subplots(len(periods), 1, figsize=(8, 3.5 * len(periods)), squeeze=False)

        for i, period_type in enumerate(periods):
            ax = axes[i][0]
            for r in res_list:
                if r["period_type"] != period_type:
                    continue

                comp = r["comparison"]
                color = comp_color_map[comp]
                label = comp.replace("__vs__", " vs ")

                ax.plot(r["lags"], r["mean_delta"], label=label, color=color, linewidth=1.5)

                sig_mask = (r["p_values"] < alpha) & (r["mean_delta"] > 0)
                if np.any(sig_mask):
                    ax.scatter(r["lags"][sig_mask], r["mean_delta"][sig_mask],
                               s=12, marker='o', color=color, edgecolor='k', linewidths=0.4, alpha=0.7)

            ax.axhline(0, linestyle="--", color="gray", linewidth=0.8)
            ax.set_title(f"{period_type.capitalize()} periods", fontsize=12)
            ax.set_xlabel("Lag (ms)")
            ax.set_ylabel("Δ Crosscorr")
            ax.legend(fontsize=8, loc='upper right', frameon=True)
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

        plt.suptitle(f"Δ Crosscorr | Monkey Pair: {monkey_pair}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        fname = f"{monkey_pair.replace('-', '_')}_crosscorr_deltas_combined.png"
        plt.savefig(plot_dir / fname, dpi=150)
        plt.close()

    print(f"All plots saved to: {plot_dir}")

