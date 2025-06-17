# src/socialgaze/features/crosscorr_calculator.py

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from scipy.signal import fftconvolve
import logging
import os
from joblib import Parallel, delayed
from contextlib import nullcontext

from socialgaze.config.crosscorr_config import CrossCorrConfig
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.utils.path_utils import get_crosscorr_output_path
from socialgaze.utils.loading_utils import load_df_from_pkl
from socialgaze.utils.saving_utils import save_df_to_pkl
from socialgaze.utils.hpc_utils import generate_crosscorr_job_file, submit_dsq_array_job

logger = logging.getLogger(__name__)


class CrossCorrCalculator:
    def __init__(
        self,
        config: CrossCorrConfig,
        fixation_detector: FixationDetector,
        interactivity_detector: Optional[InteractivityDetector] = None,
    ):
        self.config = config
        self.fixation_detector = fixation_detector
        self.interactivity_detector = interactivity_detector

    def compute_crosscorrelations(self, by_interactivity_period: bool = False):
        if by_interactivity_period:
            assert self.interactivity_detector is not None, "InteractivityDetector must be provided."
            logger.info("Computing cross-correlations by interactivity period...")
            inter_df = self.interactivity_detector.load_interactivity_periods()
        else:
            logger.info("Computing full-period cross-correlations...")

        for a1, b1, a2, b2 in tqdm(self.config.crosscorr_agent_behavior_pairs, desc="Agent-behavior pairs"):
            try:
                df1 = self.fixation_detector.get_binary_vector_df(b1)
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

                all_rows = [row for batch in results if batch for row in batch]

                if all_rows:
                    full_df = pd.concat(all_rows, ignore_index=True)
                    name = f"{a1}_{b1}__vs__{a2}_{b2}"
                    if period_type != "full":
                        name += f"_{period_type}"
                    _save_crosscorr_df(full_df, name, self.config)

        logger.info("Cross-correlation computation complete.")


    def compute_shuffled_crosscorrelations(self, by_interactivity_period: bool = False):
        """
        If `run_single_test_case` is True in config, runs one task from the full set.
        Otherwise submits a SLURM job array.
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
            import random
            test_task = random.choice(tasks)
            logger.info(f"Running single test task: {test_task}")
            self.compute_shuffled_crosscorrelations_for_single_run(*test_task)
            return

        generate_crosscorr_job_file(tasks, self.config)
        submit_dsq_array_job(self.config)



    def compute_shuffled_crosscorrelations_for_single_run(self, session, run, a1, b1, a2, b2, period_type="full"):
        try:
            df1 = self.fixation_detector.get_binary_vector_df(b1)
            df2 = self.fixation_detector.get_binary_vector_df(b2)
        except FileNotFoundError:
            logger.warning(f"Missing binary vector: {b1} or {b2}")
            return

        df1 = df1[(df1["agent"] == a1) & (df1["session_name"] == session) & (df1["run_number"] == run)]
        df2 = df2[(df2["agent"] == a2) & (df2["session_name"] == session) & (df2["run_number"] == run)]

        if df1.empty or df2.empty:
            logger.warning(f"No data for session {session}, run {run}")
            return

        run_df = pd.concat([df1, df2], ignore_index=True)
        v1, v2 = _get_vectors_for_run(run_df, a1, b1, a2, b2)
        if v1 is None or v2 is None:
            return

        if period_type == "full":
            periods = [(0, len(v1) - 1)]
        else:
            inter_df = self.interactivity_detector.load_interactivity_periods()
            periods = _get_periods_for_run(inter_df, session, run, period_type, len(v1))
            if not periods:
                return

        all_rows = []
        for start, stop in periods:
            seg1 = v1[start:stop + 1]
            seg2 = v2[start:stop + 1]
            if len(seg1) < 2 or len(seg2) < 2:
                continue

            try:
                shuffled_vecs = _generate_shuffled_vectors_for_run(
                    seg1, run_length=len(seg1),
                    num_shuffles=self.config.num_shuffles,
                    num_cpus=self.config.num_cpus,
                    stringent=self.config.make_shuffle_stringent,
                )
            except Exception as e:
                logger.warning(f"Shuffling failed for {session}-{run} | {a1} {b1}: {e}")
                continue

            corrs = Parallel(n_jobs=self.config.num_cpus)(
                delayed(_compute_normalized_crosscorr)(
                    s_v1, seg2,
                    max_lag=None,
                    normalize=self.config.normalize,
                    use_energy_norm=self.config.use_energy_norm
                )[1]  # We only need the correlation values
                for s_v1 in shuffled_vecs
            )

            corrs = np.stack(corrs)
            mean_corr = np.mean(corrs, axis=0)
            std_corr = np.std(corrs, axis=0)
            lags = _compute_normalized_crosscorr(seg1, seg2)[0]

            df = pd.DataFrame({
                "session_name": session,
                "run_number": run,
                "agent1": a1,
                "agent2": a2,
                "behavior1": b1,
                "behavior2": b2,
                "lag": lags,
                "crosscorr_mean": mean_corr,
                "crosscorr_std": std_corr,
                "period_type": period_type
            })
            all_rows.append(df)

        if all_rows:
            out_df = pd.concat(all_rows, ignore_index=True)
            name = f"{a1}_{b1}__vs__{a2}_{b2}_shuffled"
            if period_type != "full":
                name += f"_{period_type}"
            
            save_dir = self.config.crosscorr_shuffled_output_dir
            os.makedirs(save_dir, exist_ok=True)
            out_path = save_dir / f"{name}__{session}__run{run}.pkl"
            out_df.to_pickle(out_path)
            logger.info(f"Saved shuffled crosscorr: {out_path}")



    def load_crosscorr_df(self, comparison_name: str) -> pd.DataFrame:
        path = get_crosscorr_output_path(self.config, comparison_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cross-correlation result not found at: {path}")
        return load_df_from_pkl(path)


def _process_one_session_run_crosscorr(session, run, run_df, a1, b1, a2, b2, period_type, inter_df, config):
    v1, v2 = _get_vectors_for_run(run_df, a1, b1, a2, b2)
    if v1 is None or v2 is None:
        return []

    if period_type == "full":
        periods = [(0, len(v1) - 1)]
    else:
        periods = _get_periods_for_run(inter_df, session, run, period_type, len(v1))
        if not periods:
            return []

    return _compute_crosscorr_for_periods(session, run, a1, a2, b1, b2, periods, v1, v2, period_type, config)


def _get_vectors_for_run(run_df, a1, b1, a2, b2):
    try:
        agent_data = {
            (row["agent"], row["behavior_type"]): row["binary_vector"]
            for _, row in run_df.iterrows()
        }
        return agent_data[(a1, b1)], agent_data[(a2, b2)]
    except KeyError:
        return None, None

def _get_periods_for_run(inter_df, session, run, period_type, full_len):
    run_periods = inter_df[
        (inter_df.session_name == session) &
        (inter_df.run_number == run)
    ]
    if run_periods.empty:
        return []
    inter = run_periods[["start", "stop"]].values
    return inter if period_type == "interactive" else _compute_complement_periods(inter, [(0, full_len - 1)])

def _compute_crosscorr_for_periods(session, run, a1, a2, b1, b2, periods, v1, v2, period_type, config):
    all_rows = []
    for start, stop in periods:
        seg1 = v1[start:stop + 1]
        seg2 = v2[start:stop + 1]
        if len(seg1) < 2 or len(seg2) < 2:
            continue
        lags, corr = _compute_normalized_crosscorr(
            seg1, seg2,
            max_lag=None,
            normalize=config.normalize,
            use_energy_norm=config.use_energy_norm
        )
        rows = pd.DataFrame({
            "session_name": session,
            "run_number": run,
            "agent1": a1,
            "agent2": a2,
            "behavior1": b1,
            "behavior2": b2,
            "lag": lags,
            "crosscorr": corr,
            "period_type": period_type
        })
        all_rows.append(rows)
    return all_rows

def _save_crosscorr_df(df, comparison_name, config):
    path = get_crosscorr_output_path(config, comparison_name)
    os.makedirs(path.parent, exist_ok=True)
    logger.info(f"Saving {comparison_name} dataframe.")
    save_df_to_pkl(df, path)


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


def _generate_shuffled_vectors_for_run(
    original_vector: np.ndarray,
    run_length: int,
    num_shuffles: int,
    num_cpus: int,
    stringent: bool
) -> List[np.ndarray]:
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
