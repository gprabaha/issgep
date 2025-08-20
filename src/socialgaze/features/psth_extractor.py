# src/socialgaze/features/psth_extractor.py

import os
import logging
from typing import Optional, Iterable
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


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


    # ----------------------------
    # == Public compute methods ==
    # ----------------------------

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
            logger.info(f"Running PSTH extraction in parallel using {self.config.num_cpus} CPU(s)...")
            with tqdm_joblib(tqdm(total=len(tasks), desc="PSTH extraction")):
                results = Parallel(n_jobs=self.config.num_cpus)(
                    delayed(_process_session_for_psth)(*args) for args in tasks
                )
        else:
            logger.info("Running PSTH extraction in serial mode...")
            results = [_process_session_for_psth(*args) for args in tqdm(tasks, desc="PSTH extraction")]

        rows = [r for result in results for r in result]
        self.psth_per_trial = pd.DataFrame(rows)
        save_df_to_pkl(self.psth_per_trial, self.config.psth_per_trial_path)
        logger.info(f"Saved PSTH per trial to {self.config.psth_per_trial_path}")



    def compute_avg_face_obj(self):
        """Always compute and save avg_face_obj"""
        if self.psth_per_trial is None:
            logger.info("PSTH per trial not found in memory — fetching it first...")
            self.fetch_psth("trial_wise")

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
        """Always compute and save avg_int_non_int_face with category = face_interactive / face_non_interactive"""
        if self.psth_per_trial is None:
            logger.info("PSTH per trial not found in memory — fetching it first...")
            self.fetch_psth("trial_wise")

        logger.info("Computing average PSTH: face_interactive vs face_non_interactive...")

        df = self.psth_per_trial.copy()
        df = df[df["category"] == "face"]

        df["adjusted_category"] = df["is_interactive"].map(
            lambda val: "face_interactive" if val == "interactive" else "face_non_interactive"
        )

        rows = []
        grouped = df.groupby(["unit_uuid", "adjusted_category"])
        for (unit_uuid, adjusted_category), group in grouped:
            psth_array = np.stack(group["firing_rate"].apply(np.array))
            mean_psth = psth_array.mean(axis=0)
            row = group.iloc[0].to_dict()
            rows.append({
                "unit_uuid": unit_uuid,
                "category": adjusted_category,
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


    def compute_interactive_face_significance(
        self,
        alpha: float = 0.05,
        test_window: tuple[float, float] = (-0.45, 0.45),
        save: bool = True,
    ) -> pd.DataFrame:
        """
        For each unit, test interactive vs non-interactive PSTHs (face fixations only)
        with per-bin Welch t-tests (no FDR). A unit is 'significant' if any bin in the
        test window has p < alpha.
        """
        if self.psth_per_trial is None:
            self.fetch_psth("trial_wise")

        df = self.psth_per_trial
        df_face = df[df["category"] == "face"].copy()
        if df_face.empty:
            logger.warning("No face-fixation PSTHs found; skipping significance.")
            return pd.DataFrame()

        bin_size = self.config.psth_bin_size
        start_offset, end_offset = self.config.psth_window
        num_bins = int((end_offset - start_offset) / bin_size)
        time_axis = (np.linspace(start_offset, end_offset, num_bins + 1)[:-1] + bin_size / 2.0).astype(float)

        valid_bins_mask = (time_axis >= test_window[0]) & (time_axis <= test_window[1])
        valid_idx = np.where(valid_bins_mask)[0]

        rows = []
        for unit_id, g in df_face.groupby("unit_uuid"):
            region = g["region"].iloc[0] if "region" in g.columns else None
            channel = g["channel"].iloc[0] if "channel" in g.columns else None

            g_int = g[g["is_interactive"] == "interactive"]
            g_non = g[g["is_interactive"] == "non-interactive"]

            if g_int.empty or g_non.empty:
                rows.append({
                    "unit_uuid": unit_id, "region": region, "channel": channel,
                    "n_interactive": len(g_int), "n_non_interactive": len(g_non),
                    "significant": False, "significant_bins": [],
                    "time_axis": time_axis.tolist(),
                    "mean_psth_interactive": None, "mean_psth_non_interactive": None,
                })
                continue

            X = np.stack(g_int["firing_rate"].apply(np.array).values)   # (n_int, num_bins)
            Y = np.stack(g_non["firing_rate"].apply(np.array).values)   # (n_non, num_bins)
            if X.shape[1] != num_bins or Y.shape[1] != num_bins:
                logger.warning(f"Bin mismatch for unit {unit_id}; skipping.")
                rows.append({
                    "unit_uuid": unit_id, "region": region, "channel": channel,
                    "n_interactive": len(g_int), "n_non_interactive": len(g_non),
                    "significant": False, "significant_bins": [],
                    "time_axis": time_axis.tolist(),
                    "mean_psth_interactive": None, "mean_psth_non_interactive": None,
                })
                continue

            pvals = np.ones(num_bins, dtype=float)
            for b in valid_idx:
                try:
                    pvals[b] = ttest_ind(X[:, b], Y[:, b], equal_var=False, nan_policy="omit").pvalue
                except Exception as e:
                    logger.debug(f"ttest failed for unit {unit_id}, bin {b}: {e}")
                    pvals[b] = 1.0

            sig_bins = np.where((pvals < alpha) & valid_bins_mask)[0].tolist()
            is_sig = len(sig_bins) > 0

            rows.append({
                "unit_uuid": unit_id,
                "region": region,
                "channel": channel,
                "n_interactive": int(len(X)),
                "n_non_interactive": int(len(Y)),
                "significant": bool(is_sig),
                "significant_bins": sig_bins,
                "time_axis": time_axis.tolist(),
                "mean_psth_interactive": X.mean(axis=0).tolist(),
                "mean_psth_non_interactive": Y.mean(axis=0).tolist(),
            })

        sig_df = pd.DataFrame(rows)

        if save:
            out_dir = Path(self.config.output_dir) / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "interactive_face_significance.pkl"
            save_df_to_pkl(sig_df, str(out_path))
            logger.info(f"Saved interactive face significance to {out_path}")

        return sig_df


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




class PSTHPlotter(PSTHExtractor):
    """
    Plots:
      1) Unit PSTHs for significant interactive vs non-interactive (face) units
      2) Per-region pie charts (significant vs non-significant)
    """

    def plot_significant_interactive_vs_noninteractive_units(
        self,
        alpha: float = 0.05,
        test_window: tuple[float, float] = (-0.45, 0.45),
        export_formats: tuple[str, ...] = ("pdf",),
        overwrite: bool = False,
    ) -> pd.DataFrame:
        sig_df = self._load_or_compute_significance(alpha=alpha, test_window=test_window)
        if sig_df.empty:
            logger.warning("No significance results available; nothing to plot.")
            return sig_df

        base_plot_dir = Path(getattr(self.config, "plots_dir", getattr(self.config, "plot_dir", ".")))
        unit_out_root = base_plot_dir / "psth" / "interactive_units"
        unit_out_root.mkdir(parents=True, exist_ok=True)

        for region, region_df in sig_df.groupby("region", dropna=False):
            region_name = str(region) if pd.notna(region) else "UnknownRegion"
            region_dir = unit_out_root / region_name
            region_dir.mkdir(parents=True, exist_ok=True)

            # Unit PSTHs for significant units
            sig_units = region_df[region_df["significant"] == True]
            if not sig_units.empty:
                logger.info(f"[{region_name}] Plotting {len(sig_units)} significant unit PSTHs...")
                for _, row in sig_units.iterrows():
                    self._plot_single_unit_psth(row=row, region_dir=region_dir,
                                                export_formats=export_formats, overwrite=overwrite)
            else:
                logger.info(f"[{region_name}] No significant units; skipping PSTH plots.")

            # Region pie
            self._plot_region_significance_pie(
                region=region_name,
                region_df=region_df,
                out_dir=region_dir,
                export_formats=export_formats,
                overwrite=overwrite,
            )

        return sig_df

    # -------- helpers --------

    def _load_or_compute_significance(
        self,
        alpha: float,
        test_window: tuple[float, float],
    ) -> pd.DataFrame:
        sig_path = Path(self.config.output_dir) / "results" / "interactive_face_significance.pkl"
        if sig_path.exists():
            try:
                sig_df = load_df_from_pkl(str(sig_path))
                needed = {"unit_uuid", "region", "significant", "time_axis",
                          "mean_psth_interactive", "mean_psth_non_interactive"}
                if needed.issubset(sig_df.columns):
                    logger.info(f"Loaded significance from {sig_path}")
                    return sig_df
                logger.warning(f"Significance file missing columns; recomputing.")
            except Exception as e:
                logger.warning(f"Failed to load significance file: {e}; recomputing.")

        return self.compute_interactive_face_significance(alpha=alpha, test_window=test_window, save=True)

    def _plot_single_unit_psth(
        self,
        row: pd.Series,
        region_dir: Path,
        export_formats: tuple[str, ...],
        overwrite: bool,
    ) -> None:
        unit_id = row["unit_uuid"]
        time_axis = np.asarray(row["time_axis"], dtype=float)
        m_int = row["mean_psth_interactive"]
        m_non = row["mean_psth_non_interactive"]
        if m_int is None or m_non is None:
            return
        m_int = np.asarray(m_int, dtype=float)
        m_non = np.asarray(m_non, dtype=float)
        sig_bins = row["significant_bins"] or []

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time_axis, m_int, label="Interactive", linewidth=2)
        ax.plot(time_axis, m_non, label="Non-interactive", linewidth=2, linestyle="--")

        if len(sig_bins) > 0:
            spans = _bins_to_spans(sig_bins, time_axis)
            for (t0, t1) in spans:
                ax.axvspan(t0, t1, alpha=0.15)

        ax.axvline(0.0, linestyle=":", linewidth=1)
        ax.set_xlabel("Time from fixation onset (s)")
        ax.set_ylabel("Firing rate (spikes/s)")
        ax.set_title(f"Unit {unit_id}")
        ax.legend(loc="upper right")
        ax.margins(x=0)

        for ext in export_formats:
            out_path = region_dir / f"unit_{unit_id}.{ext}"
            if overwrite or not out_path.exists():
                fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_region_significance_pie(
        self,
        region: str,
        region_df: pd.DataFrame,
        out_dir: Path,
        export_formats: tuple[str, ...],
        overwrite: bool,
    ) -> None:
        n_sig = int(region_df["significant"].sum())
        n_total = int(len(region_df))
        n_non = n_total - n_sig

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.pie([n_sig, n_non], labels=["Significant", "Not significant"], autopct="%1.0f%%", startangle=90)
        ax.set_title(f"{region}: Interactive vs Non-interactive (face)")

        for ext in export_formats:
            out_path = out_dir / f"{region}_interactive_face_signif_pie.{ext}"
            if overwrite or not out_path.exists():
                fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def _bins_to_spans(bin_indices: Iterable[int], time_axis: np.ndarray) -> list[tuple[float, float]]:
    """Convert discrete significant bin indices to contiguous time spans (bin-center based)."""
    if not bin_indices:
        return []
    bin_indices = np.array(sorted(bin_indices), dtype=int)
    diffs = np.diff(bin_indices)
    run_starts = np.r_[0, np.where(diffs != 1)[0] + 1]
    run_ends = np.r_[run_starts[1:] - 1, len(bin_indices) - 1]
    if len(time_axis) > 1:
        half = np.median(np.diff(time_axis)) / 2.0
    else:
        half = 0.0
    spans = []
    for s, e in zip(run_starts, run_ends):
        i0, i1 = bin_indices[s], bin_indices[e]
        spans.append((float(time_axis[i0] - half), float(time_axis[i1] + half)))
    return spans
