# src/socialgaze/features/crosscorr_calculator.py

import pdb
import logging
import os
from typing import Optional, Optional, List, Tuple
from collections import defaultdict

import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_1samp
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib import cm

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
            # self.combine_and_save_shuffled_results()
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
            np.zeros_like(v1),
            np.zeros_like(v2),
            config
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


    def analyze_crosscorr_vs_shuffled_per_pair(self, analysis_strategy: str = None):
        VALID_STRATEGIES = ["default", "by_dominance", "by_leader_follower"]

        session_to_monkey_pair = self.config.ephys_days_and_monkeys_df.set_index("session_name")[["m1", "m2"]].to_dict("index")
        dominance_lookup = {
            f"{row['Monkey Pair']}": row["dominant_agent_label"]
            for _, row in self.config.monkey_dominance_df.iterrows()
        }
        

        expanded_rows = []
        for a1, b1, a2, b2 in self.config.crosscorr_agent_behavior_pairs:
            for period_type in ["full", "interactive", "non_interactive"]:
                expanded_rows.extend(
                    self._process_crosscorr_pair(
                        a1, b1, a2, b2, period_type, session_to_monkey_pair, dominance_lookup
                    )
                )
        
        df = pd.DataFrame(expanded_rows)
        df = _assign_leaders_by_crosscorr(df)
        df = _assign_dominance_direction(df)
        
        strategies_to_run = VALID_STRATEGIES if analysis_strategy is None else [analysis_strategy]
        for strategy in strategies_to_run:
            result_df = _aggregate_and_test(df, strategy)
            self.save_crosscorr_analysis_results(result_df, strategy=strategy)


    def _process_crosscorr_pair(self, a1, b1, a2, b2, period_type, session_to_monkey_pair, dominance_lookup):
        obs_path = self.paths.get_obs_crosscorr_path(a1, b1, a2, b2, period_type)
        shuffled_path = self.paths.get_shuffled_final_path(a1, b1, a2, b2, period_type)

        if not obs_path.exists() or not shuffled_path.exists():
            logger.warning(f"Missing files for {a1}-{b1} vs {a2}-{b2} [{period_type}]")
            return []

        try:
            observed_df = pd.read_pickle(obs_path)
            shuffled_df = pd.read_pickle(shuffled_path)
        except Exception as e:
            logger.warning(f"Failed to load data for {a1}-{b1} vs {a2}-{b2}: {e}")
            return []

        rows = []
        for (session, run), obs_group in observed_df.groupby(["session_name", "run_number"]):
            run = str(run)
            session = str(session)
            monkey_ids = session_to_monkey_pair.get(session)
            if not monkey_ids:
                continue

            try:
                shuffle_row = shuffled_df.query("session_name == @session and run_number == @run").iloc[0]
            except IndexError:
                continue

            m1, m2 = monkey_ids["m1"], monkey_ids["m2"]
            monkey_pair = f"{m1} vs {m2}"
            dominant = dominance_lookup.get(monkey_pair)

            lags = obs_group.iloc[0]["lags"]
            obs_corr = obs_group.iloc[0]["crosscorr"]
            shuffled_mean = shuffle_row["crosscorr_mean"]
            delta = obs_corr - shuffled_mean

            pos_mask = lags >= 0
            neg_mask = lags <= 0

            rows.append({
                "session": session, "run": run, "monkey_pair": monkey_pair, "period_type": period_type,
                "a1": a1, "b1": b1, "a2": a2, "b2": b2, "m1": m1, "m2": m2,
                "effective_direction": f"{a1}_{b1} → {a2}_{b2}",
                "lag_direction": "positive_lags", "lags": lags[pos_mask], "delta": delta[pos_mask],
                "monkey_dominant": dominant
            })

            rows.append({
                "session": session, "run": run, "monkey_pair": monkey_pair, "period_type": period_type,
                "a1": a1, "b1": b1, "a2": a2, "b2": b2, "m1": m1, "m2": m2,
                "effective_direction": f"{a2}_{b2} → {a1}_{b1}",
                "lag_direction": "negative_lags_flipped", "lags": -lags[neg_mask][::-1], "delta": delta[neg_mask][::-1],
                "monkey_dominant": dominant
            })
        return rows


    def save_crosscorr_analysis_results(self, results: pd.DataFrame, strategy: str):
        """
        Saves results to: results/mean_minus_shuffled_crosscorr_results_<strategy>.pkl
        """
        out_path = self.paths.get_analysis_output_path(strategy=strategy)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(results, out_path)
        print(f"Saved delta crosscorr results to {out_path}")


    def plot_crosscorr_deltas(self, alpha=0.05):
        plot_dir = self.config.paths.get_crosscorr_deltas_plot_dir()
        strategies = ["default", "by_dominance", "by_leader_follower"]

        for strategy in strategies:
            result_path = self.paths.get_analysis_output_path(strategy=strategy)
            if not result_path.exists():
                logger.warning(f"Result file not found for strategy: {strategy}")
                continue

            df = pd.read_pickle(result_path)
            _make_crosscorr_deltas_plot(df, strategy, plot_dir, alpha=alpha)


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
        config
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


def _compute_normalized_crosscorr(
    x: np.ndarray,
    y: np.ndarray,
    config
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes normalized cross-correlation between x and y.
    Parameters are controlled via config:
      - config.max_lag
      - config.normalize
      - config.use_energy_norm
      - config.do_smoothing
      - config.smoothing_sigma_n_bins
    """
    x = x.astype(float)
    y = y.astype(float)

    # --- Optional smoothing ---
    if getattr(config, "do_smoothing", False):
        sigma = getattr(config, "smoothing_sigma_n_bins", 1)
        x = gaussian_filter1d(x, sigma=sigma, mode='reflect')
        y = gaussian_filter1d(y, sigma=sigma, mode='reflect')

    # --- Cross-correlation ---
    corr_full = fftconvolve(x, y[::-1], mode="full")
    lags_full = np.arange(-len(y) + 1, len(x))
    corr = corr_full
    lags = lags_full

    # --- Lag limit ---
    max_lag = getattr(config, "max_lag", None)
    if max_lag is not None:
        center = len(corr_full) // 2
        corr = corr_full[center - max_lag:center + max_lag + 1]
        lags = lags_full[center - max_lag:center + max_lag + 1]

    # --- Normalization ---
    if getattr(config, "normalize", True):
        if getattr(config, "use_energy_norm", True):
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
        config
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


# ---------------------
# Data analysis helpers
# ---------------------

def _assign_leaders_by_crosscorr(df):
    df["avg_delta"] = df["delta"].apply(np.mean)

    # Prepare a new column for assignment
    df["leader_based_direction"] = "unknown"

    # Get unique group keys
    group_keys = df.groupby(["monkey_pair", "period_type", "a1", "b1", "a2", "b2"]).groups.keys()

    for key in group_keys:
        monkey_pair, period_type, a1, b1, a2, b2 = key
        mask = (
            (df["monkey_pair"] == monkey_pair) &
            (df["period_type"] == period_type) &
            (df["a1"] == a1) & (df["b1"] == b1) &
            (df["a2"] == a2) & (df["b2"] == b2)
        )

        df_subset = df[mask]
        pos_mean = df_subset[df_subset["lag_direction"] == "positive_lags"]["avg_delta"].mean()
        neg_mean = df_subset[df_subset["lag_direction"] == "negative_lags_flipped"]["avg_delta"].mean()

        if pd.isna(pos_mean) or pd.isna(neg_mean):
            label_pos, label_neg = "unknown", "unknown"
        elif pos_mean > neg_mean:
            label_pos, label_neg = "leader_to_follower", "follower_to_leader"
        else:
            label_pos, label_neg = "follower_to_leader", "leader_to_follower"

        df.loc[mask & (df["lag_direction"] == "positive_lags"), "leader_based_direction"] = label_pos
        df.loc[mask & (df["lag_direction"] == "negative_lags_flipped"), "leader_based_direction"] = label_neg

    return df


def _assign_dominance_direction(df):
    labels = []
    for _, row in df.iterrows():
        dom = row["monkey_dominant"]
        if dom is None or pd.isna(dom):
            labels.append("unknown")
        elif row["lag_direction"] == "positive_lags":
            labels.append("dominant_to_recessive" if row["a1"] == dom else "recessive_to_dominant")
        else:
            labels.append("dominant_to_recessive" if row["a2"] == dom else "recessive_to_dominant")
    df["dominance_based_direction"] = labels
    return df


def _aggregate_and_test(df, strategy):
    if strategy == "default":
        grouping_col = "effective_direction"
        filter_df = df
    elif strategy == "by_dominance":
        grouping_col = "dominance_based_direction"
        filter_df = df[df[grouping_col].isin(["dominant_to_recessive", "recessive_to_dominant"])]
    elif strategy == "by_leader_follower":
        grouping_col = "leader_based_direction"
        filter_df = df[df[grouping_col].isin(["leader_to_follower", "follower_to_leader"])]
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    result_rows = []

    # Per-monkey-pair aggregation
    grouped = filter_df.groupby(["monkey_pair", "period_type", grouping_col, "a1", "b1", "a2", "b2"])
    for key, group in grouped:
        monkey_pair, period_type, direction_label, a1, b1, a2, b2 = key
        lags = group.iloc[0]["lags"]
        deltas = np.vstack(group["delta"].to_numpy())
        mean_delta = deltas.mean(axis=0)
        _, p_vals = ttest_1samp(deltas, popmean=0, axis=0, alternative="greater")

        rep = group.iloc[0]

        result_rows.append({
            "monkey_pair": monkey_pair,
            "period_type": period_type,
            "direction_label": direction_label,
            "effective_direction": rep["effective_direction"],
            "lags": lags,
            "mean_delta": mean_delta,
            "p_values": p_vals,
            "n_runs": len(group),
            "a1": a1,
            "a2": a2,
            "b1": b1,
            "b2": b2,
            "m1": rep["m1"],
            "m2": rep["m2"],
            "monkey_dominant": rep.get("monkey_dominant", None)
        })

    # Global aggregation across all monkey pairs
    grouped_all = filter_df.groupby([grouping_col, "period_type", "a1", "b1", "a2", "b2"])
    for key, group in grouped_all:
        direction_label, period_type, a1, b1, a2, b2 = key
        if group.empty:
            continue
        lags = group.iloc[0]["lags"]
        deltas = np.vstack(group["delta"].to_numpy())
        mean_delta = deltas.mean(axis=0)
        _, p_vals = ttest_1samp(deltas, popmean=0, axis=0, alternative="greater")

        rep = group.iloc[0]

        result_rows.append({
            "monkey_pair": "ALL",
            "period_type": period_type,
            "direction_label": direction_label,
            "effective_direction": rep["effective_direction"],
            "lags": lags,
            "mean_delta": mean_delta,
            "p_values": p_vals,
            "n_runs": len(group),
            "a1": a1,
            "a2": a2,
            "b1": b1,
            "b2": b2,
            "m1": rep["m1"],
            "m2": rep["m2"],
            "monkey_dominant": rep.get("monkey_dominant", None)
        })

    return pd.DataFrame(result_rows)


# ----------
# Plotting
# -----------

# Re-import necessary modules after code execution environment reset
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label
from matplotlib.lines import Line2D
from itertools import cycle


def _plot_crosscorr_with_significance(ax, lags, mean_delta, p_values, alpha, color, label_name):
    """Plot cross-correlation with non-significant and significant portions separated."""
    sig_mask = p_values < alpha

    # Plot full line (low alpha)
    ax.plot(lags, mean_delta, linewidth=1.0, alpha=0.3, color=color, label=f"{label_name} (all)")

    # Find contiguous regions of significance and plot separately with higher alpha and linewidth
    labeled_array, num_features = label(sig_mask)
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        ax.plot(lags[indices], mean_delta[indices], linewidth=2.0, alpha=1.0, color=color, label=f"{label_name} (sig)" if i == 1 else None)


def _make_crosscorr_deltas_plot(df, strategy, plot_dir, alpha=0.05, max_lag_ms=10000):
    """Generate and save plots for cross-correlation deltas for a specific strategy."""
    df = df[df["lags"].apply(lambda x: np.max(np.abs(x)) >= max_lag_ms)]
    strategy_dir = Path(plot_dir) / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)

    all_keys = df["monkey_pair"].unique()

    # Color mapping for different direction labels
    unique_directions = df["direction_label"].unique()
    color_cycle = cycle(plt.cm.tab10.colors)
    direction_to_color = {label: next(color_cycle) for label in sorted(unique_directions)}

    for monkey_pair in all_keys:
        df_pair = df[df["monkey_pair"] == monkey_pair]
        if df_pair.empty:
            continue

        period_types = sorted(df_pair["period_type"].unique())
        ab_combos = df_pair[["a1", "b1", "a2", "b2"]].drop_duplicates().to_records(index=False)

        n_rows = len(period_types)
        n_cols = len(ab_combos)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False)
        fig.subplots_adjust(hspace=0.35, wspace=0.25)

        for i, period in enumerate(period_types):
            for j, (a1, b1, a2, b2) in enumerate(ab_combos):
                ax = axs[i][j]
                df_subset = df_pair[
                    (df_pair["period_type"] == period)
                    & (df_pair["a1"] == a1) & (df_pair["b1"] == b1)
                    & (df_pair["a2"] == a2) & (df_pair["b2"] == b2)
                ]

                if df_subset.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                    continue

                # Plot each direction separately
                for k, label_direction in enumerate(sorted(df_subset["direction_label"].unique())):
                    row = df_subset[df_subset["direction_label"] == label_direction].iloc[0]
                    lags = row["lags"]
                    mask = np.abs(lags) <= max_lag_ms

                    _plot_crosscorr_with_significance(
                        ax,
                        lags[mask],
                        row["mean_delta"][mask],
                        row["p_values"][mask],
                        alpha=alpha,
                        color=direction_to_color[label_direction],
                        label_name=label_direction if i == 0 else None  # Show label only in top row
                    )

                ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
                if i == n_rows - 1:
                    ax.set_xlabel("Lag (ms)")
                if j == 0:
                    ax.set_ylabel("Δ correlation")

        # Add row labels (period types)
        for row_idx, period in enumerate(period_types):
            axs[row_idx][0].annotate(period, xy=(-0.4, 0.5), xycoords='axes fraction',
                                     fontsize=12, ha='center', va='center', rotation=90)

        # Add column labels (a1-b1 vs a2-b2)
        for col_idx, (a1, b1, a2, b2) in enumerate(ab_combos):
            col_label = f"{a1}-{b1}\nvs\n{a2}-{b2}"
            axs[0][col_idx].set_title(col_label, fontsize=11)

        m1, m2 = df_pair.iloc[0]["m1"], df_pair.iloc[0]["m2"]
        dom = df_pair.iloc[0].get("monkey_dominant", "N/A")

        if monkey_pair != "ALL":
            title = f"{monkey_pair} | m1: {m1}, m2: {m2} | Dominant: {dom}"
        else:
            title = "Averaged Across All Pairs"

        fig.suptitle(f"Crosscorr Δ (Observed - Shuffled) [{strategy}] | {title}", fontsize=14)

        # Add legend manually to the right
        handles = [Line2D([0], [0], color=color, lw=2, label=label)
                   for label, color in direction_to_color.items()]
        fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.02, 1.0), fontsize=10)

        save_path = strategy_dir / f"{monkey_pair.replace(' ', '_')}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
