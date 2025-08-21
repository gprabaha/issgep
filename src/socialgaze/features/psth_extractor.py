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
from matplotlib import rc_context
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm



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
                logger.info(f"Loading PSTH per trial from {path}")
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
        test_window: tuple[float, float] = (-0.5, 0.5),
        save: bool = True,
    ) -> pd.DataFrame:
        """
        For each unit, test interactive vs non-interactive PSTHs (face fixations only)
        with per-bin Welch t-tests (no FDR).

        Significance rule:
          - significant if (longest run of consecutive significant bins) >= min_consecutive_sig_bins
            OR (total significant bins) >= min_total_sig_bins.

        Thresholds are fetched from config with defaults:
          - min_consecutive_sig_bins (default 5)
          - min_total_sig_bins (default 25)
        """
        from tqdm import tqdm

        # Thresholds from config (with defaults)
        min_consec = int(getattr(self.config, "min_consecutive_sig_bins", 5))
        min_total  = int(getattr(self.config, "min_total_sig_bins", 25))

        if self.psth_per_trial is None:
            self.fetch_psth("trial_wise")

        df = self.psth_per_trial
        df_face = df[df["category"] == "face"].copy()
        if df_face.empty:
            logger.warning("No face-fixation PSTHs found; skipping significance.")
            return pd.DataFrame()

        # Build time axis
        bin_size = self.config.psth_bin_size
        start_offset, end_offset = self.config.psth_window
        num_bins = int((end_offset - start_offset) / bin_size)
        time_axis = (np.linspace(start_offset, end_offset, num_bins + 1)[:-1] + bin_size / 2.0).astype(float)

        valid_bins_mask = (time_axis >= test_window[0]) & (time_axis <= test_window[1])
        valid_idx = np.where(valid_bins_mask)[0]

        n_units = df_face["unit_uuid"].nunique()
        logger.info(
            f"Interactive-vs-NonInteractive (face) significance:\n"
            f"  alpha={alpha}, window={test_window}, bins={num_bins}\n"
            f"  thresholds: min_consecutive={min_consec}, min_total={min_total}\n"
            f"  units to test: {n_units}"
        )

        rows = []
        sig_count = 0

        for unit_id, g in tqdm(df_face.groupby("unit_uuid"), total=n_units, desc="Testing units (face)"):
            try:
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

                X = np.stack(g_int["firing_rate"].apply(np.array).values)
                Y = np.stack(g_non["firing_rate"].apply(np.array).values)
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

                # Per-bin Welch t-test on valid window
                pvals = np.ones(num_bins, dtype=float)
                for b in valid_idx:
                    try:
                        pvals[b] = ttest_ind(X[:, b], Y[:, b], equal_var=False, nan_policy="omit").pvalue
                    except Exception as e:
                        logger.debug(f"ttest failed for unit {unit_id}, bin {b}: {e}")
                        pvals[b] = 1.0

                sig_bins = np.where((pvals < alpha) & valid_bins_mask)[0].tolist()
                is_sig = self._is_unit_significant(sig_bins, min_consec=min_consec, min_total=min_total)
                if is_sig:
                    sig_count += 1

                rows.append({
                    "unit_uuid": unit_id,
                    "region": region,
                    "channel": channel,
                    "n_interactive": int(X.shape[0]),
                    "n_non_interactive": int(Y.shape[0]),
                    "significant": bool(is_sig),
                    "significant_bins": sig_bins,
                    "time_axis": time_axis.tolist(),
                    "mean_psth_interactive": X.mean(axis=0).tolist(),
                    "mean_psth_non_interactive": Y.mean(axis=0).tolist(),
                })

            except Exception as e:
                logger.exception(f"Error while processing unit {unit_id}: {e}")
                rows.append({
                    "unit_uuid": unit_id,
                    "region": g["region"].iloc[0] if "region" in g.columns else None,
                    "channel": g["channel"].iloc[0] if "channel" in g.columns else None,
                    "n_interactive": int(g[g['is_interactive'] == 'interactive'].shape[0]),
                    "n_non_interactive": int(g[g['is_interactive'] == 'non-interactive'].shape[0]),
                    "significant": False, "significant_bins": [],
                    "time_axis": time_axis.tolist(),
                    "mean_psth_interactive": None, "mean_psth_non_interactive": None,
                })

        sig_df = pd.DataFrame(rows)
        logger.info(f"Significance complete: {sig_count} / {n_units} units marked significant.")

        if save:
            out_dir = Path(self.config.output_dir) / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "interactive_face_significance.pkl"
            save_df_to_pkl(sig_df, out_path)
            logger.info(f"Saved interactive face significance to {out_path}")

        return sig_df



    @staticmethod
    def _is_unit_significant(sig_bins: list[int], min_consec: int = 5, min_total: int = 25) -> bool:
        """
        Decide if a unit is significant based on its significant bins.

        A unit is significant if:
          - it has at least `min_consec` consecutive significant bins, OR
          - it has at least `min_total` total significant bins.
        """
        if not sig_bins:
            return False

        arr = np.array(sorted(sig_bins), dtype=int)
        if arr.size == 1:
            longest_run = 1
        else:
            diffs = np.diff(arr) == 1
            longest_run = 1
            curr = 1
            for contig in diffs:
                if contig:
                    curr += 1
                else:
                    if curr > longest_run:
                        longest_run = curr
                    curr = 1
            if curr > longest_run:
                longest_run = curr

        return (longest_run >= min_consec) or (len(sig_bins) >= min_total)



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









## ----------------
## PSTH Plotter ##
## ----------------

class PSTHPlotter(PSTHExtractor):
    """
    Plots:
      1) Unit PSTHs for significant interactive vs non-interactive (face) units
      2) Per-region pie charts (significant vs non-significant)

    Exports fully vector PDFs that are editable in Illustrator.
    """

    # Illustrator-friendly rc settings (used with matplotlib.rc_context)
    _ILLUSTRATOR_RC = {
        "pdf.fonttype": 42,        # Embed TrueType fonts (editable text)
        "ps.fonttype": 42,
        "svg.fonttype": "none",    # If exporting SVGs too
        "path.simplify": False,    # Preserve exact paths
        "savefig.transparent": False,
    }

    def plot_significant_interactive_vs_noninteractive_units(
        self,
        alpha: float = 0.05,
        test_window: tuple[float, float] = (-0.5, 0.5),
        export_formats: tuple[str, ...] = ("pdf",),
        overwrite: bool = True,
        illustrator_friendly: bool = True,
    ) -> pd.DataFrame:
        """
        Make unit-level PSTH plots for significant units and per-region pie charts.

        Plots are saved under:
            config.plots_dir / "psth" / "interactive_units" / <REGION> / ...

        Significance table is read from (or computed/saved to):
            config.output_dir / "results" / "interactive_face_significance.pkl"
        """
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
                    self._plot_single_unit_psth(
                        row=row,
                        region_dir=region_dir,
                        export_formats=export_formats,
                        overwrite=overwrite,
                        illustrator_friendly=illustrator_friendly,
                    )
            else:
                logger.info(f"[{region_name}] No significant units; skipping PSTH plots.")

            # Region pie
            self._plot_region_significance_pie(
                region=region_name,
                region_df=region_df,
                out_dir=region_dir,
                export_formats=export_formats,
                overwrite=overwrite,
                illustrator_friendly=illustrator_friendly,
            )

        return sig_df

    # ------------- helpers -------------

    def _load_or_compute_significance(
        self,
        alpha: float,
        test_window: tuple[float, float],
    ) -> pd.DataFrame:
        """
        Load cached significance if present; otherwise compute using parent method.
        """
        sig_path = Path(self.config.output_dir) / "results" / "interactive_face_significance.pkl"
        if sig_path.exists():
            try:
                sig_df = load_df_from_pkl(sig_path)
                needed = {
                    "unit_uuid", "region", "significant", "time_axis",
                    "mean_psth_interactive", "mean_psth_non_interactive"
                }
                if needed.issubset(sig_df.columns):
                    logger.info(f"Loaded significance from {sig_path}")
                    return sig_df
                logger.warning(f"Significance file {sig_path} missing columns; recomputing.")
            except Exception as e:
                logger.warning(f"Failed to load significance file {sig_path}: {e}; recomputing.")

        # Compute + save to config.output_dir / results
        return self.compute_interactive_face_significance(alpha=alpha, test_window=test_window, save=True)

    def _plot_single_unit_psth(
        self,
        row: pd.Series,
        region_dir: Path,
        export_formats: tuple[str, ...],
        overwrite: bool,
        illustrator_friendly: bool = True,
    ) -> None:
        """
        Plot mean PSTH of interactive vs non-interactive for a single significant unit.
        Shade significant bins using opaque rectangles (no alpha) to keep PDF vectorized.
        """
        unit_id = row["unit_uuid"]
        time_axis = np.asarray(row["time_axis"], dtype=float)
        m_int = row["mean_psth_interactive"]
        m_non = row["mean_psth_non_interactive"]
        if m_int is None or m_non is None:
            return
        m_int = np.asarray(m_int, dtype=float)
        m_non = np.asarray(m_non, dtype=float)
        sig_bins = row["significant_bins"] or []

        ctx = rc_context(self._ILLUSTRATOR_RC) if illustrator_friendly else rc_context()
        with ctx:
            fig, ax = plt.subplots(figsize=(6, 4))

            # Shade significant spans with opaque rectangles behind lines (no transparency)
            if len(sig_bins) > 0:
                spans = self._bins_to_spans(sig_bins, time_axis)
                ymin = float(min(m_int.min(), m_non.min()))
                ymax = float(max(m_int.max(), m_non.max()))
                for (t0, t1) in spans:
                    rect = Rectangle(
                        (t0, ymin),
                        width=(t1 - t0),
                        height=(ymax - ymin),
                        facecolor=(0.88, 0.88, 0.88),
                        edgecolor="none",
                        zorder=0,
                    )
                    ax.add_patch(rect)

            # Editable vector strokes
            ax.plot(
                time_axis, m_int,
                label="Interactive",
                linewidth=2,
                solid_joinstyle="round",
                solid_capstyle="round",
                antialiased=True,
                zorder=2,
            )
            ax.plot(
                time_axis, m_non,
                label="Non-interactive",
                linewidth=2,
                linestyle="--",
                solid_joinstyle="round",
                solid_capstyle="round",
                antialiased=True,
                zorder=2,
            )

            ax.axvline(0.0, linestyle=":", linewidth=1, zorder=1)
            ax.set_xlabel("Time from fixation onset (s)")
            ax.set_ylabel("Firing rate (spikes/s)")
            ax.set_title(f"Unit {unit_id}")
            ax.legend(loc="upper right")
            ax.margins(x=0)
            ax.set_xlim(-0.5, 0.5)

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
        illustrator_friendly: bool = True,
    ) -> None:
        """
        Pie chart: fraction and count of significant vs non-significant units in this region.
        Shows labels as: "Significant (X/Y, ZZ%)" and "Not significant (N/Y, ZZ%)".
        The 'Significant' slice is separated (exploded) for emphasis.
        """
        n_total = int(len(region_df))
        n_sig = int(region_df["significant"].sum()) if n_total > 0 else 0
        n_non = n_total - n_sig
        pct_sig = (100.0 * n_sig / n_total) if n_total > 0 else 0.0
        pct_non = 100.0 - pct_sig if n_total > 0 else 0.0

        # Labels include counts and percentages
        labels = [
            f"Significant ({n_sig}/{n_total}, {pct_sig:.0f}%)",
            f"Not significant ({n_non}/{n_total}, {pct_non:.0f}%)",
        ]

        # Explode the 'Significant' slice slightly
        explode = (0.08, 0.0) if n_sig > 0 else (0.0, 0.0)

        ctx = rc_context(self._ILLUSTRATOR_RC) if illustrator_friendly else rc_context()
        with ctx:
            fig, ax = plt.subplots(figsize=(4.5, 4.5))

            if n_total == 0:
                ax.text(0.5, 0.5, f"{region}\nNo units", ha="center", va="center")
                ax.axis("off")
            else:
                ax.pie(
                    [n_sig, n_non],
                    labels=labels,
                    startangle=90,
                    explode=explode,
                    autopct=None,                   # percentages are already in labels
                    wedgeprops=dict(linewidth=0.5), # thin editable edges
                    textprops=dict(),               # keep as text (editable)
                )
                ax.set_title(f"{region}: Interactive vs Non-interactive (face)")

            for ext in export_formats:
                out_path = out_dir / f"{region}_interactive_face_signif_pie.{ext}"
                if overwrite or not out_path.exists():
                    fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)


    # Keep it self-contained: convert discrete significant bin indices into contiguous spans
    @staticmethod
    def _bins_to_spans(bin_indices: Iterable[int], time_axis: np.ndarray) -> list[tuple[float, float]]:
        """
        Convert discrete bin indices to contiguous (start_time, end_time) spans.
        Assumes `time_axis` are bin centers; spans extend half-bin to either side.
        """
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


    def plot_region_heatmaps_of_sig_units(
        self,
        export_formats: tuple[str, ...] = ("pdf",),
        overwrite: bool = True,
        illustrator_friendly: bool = True,
        time_window: tuple[float, float] = (-0.5, 0.5),
        outfile_basename: str = "region_sig_unit_index_heatmaps",
    ) -> None:
        """
        One-row figure: a panel per region with the index heatmap:
            (interactive - noninteractive) / (interactive + noninteractive)
        for SIGNIFICANT units within `time_window`.

        Rows = units (sorted by argmax time per unit, early → late; earlier on top)
        Columns = time bins
        Top:   column sums
        Right: row sums (no y-ticks)
        Titles: region name with n_sig / n_total units
        """
        from matplotlib.colors import TwoSlopeNorm
        import matplotlib.gridspec as gridspec

        sig_df = self._load_or_compute_significance(alpha=0.05, test_window=time_window)
        if sig_df.empty:
            logger.warning("No significance results available; skipping heatmap plotting.")
            return

        # Keep both sig and non-sig for denominator of "total"
        all_df = sig_df.copy()
        sig_df = sig_df[sig_df["significant"] == True].copy()
        if sig_df.empty:
            logger.info("No significant units to plot.")
            return

        region_mats = {}
        global_min, global_max = np.inf, -np.inf

        regions = [r for r in all_df["region"].dropna().unique().tolist()]
        if len(regions) == 0:
            regions = ["UnknownRegion"]

        for region in regions:
            total_units = len(all_df[all_df["region"] == region]) if region != "UnknownRegion" else len(all_df[all_df["region"].isna()])
            rdf = sig_df[sig_df["region"] == region] if region != "UnknownRegion" else sig_df[sig_df["region"].isna()]
            if rdf.empty:
                continue

            mats, unit_ids, time_centers = [], [], None
            for _, row in rdf.iterrows():
                t = np.asarray(row["time_axis"], dtype=float)
                m_int = np.asarray(row["mean_psth_interactive"], dtype=float)
                m_non = np.asarray(row["mean_psth_non_interactive"], dtype=float)
                if m_int is None or m_non is None:
                    continue

                mask = (t >= time_window[0]) & (t <= time_window[1])
                if not mask.any():
                    continue
                t_win = t[mask]
                x = m_int[mask]
                y = m_non[mask]

                denom = x + y
                idx = np.zeros_like(denom, dtype=float)
                nz = denom != 0
                idx[nz] = (x[nz] - y[nz]) / denom[nz]

                mats.append(idx)
                unit_ids.append(row["unit_uuid"])
                time_centers = t_win

            if len(mats) == 0:
                continue

            M = np.vstack(mats)
            sort_keys = np.argmax(M, axis=1)
            order = np.argsort(sort_keys, kind="stable")[::-1]  # reversed → early maxima at top
            M = M[order]
            unit_ids = [unit_ids[i] for i in order]

            global_min = min(global_min, float(M.min()))
            global_max = max(global_max, float(M.max()))

            region_mats[region] = (M, time_centers, unit_ids, total_units)

        if not region_mats:
            logger.info("No region matrices constructed; nothing to plot.")
            return

        vmax = max(abs(global_min), abs(global_max))
        vmin = -vmax
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        base_plot_dir = Path(getattr(self.config, "plots_dir", getattr(self.config, "plot_dir", ".")))
        out_dir = base_plot_dir / "psth" / "interactive_units"
        out_dir.mkdir(parents=True, exist_ok=True)

        ctx = rc_context(self._ILLUSTRATOR_RC) if illustrator_friendly else rc_context()
        with ctx:
            nR = len(regions)
            fig = plt.figure(figsize=(6 * nR, 5.0))
            outer = gridspec.GridSpec(nrows=1, ncols=nR, wspace=0.35, hspace=0.0, figure=fig)

            mappables = []
            for ci, region in enumerate(regions):
                if region not in region_mats:
                    continue
                M, t_centers, unit_ids, total_units = region_mats[region]
                n_units, _ = M.shape

                dt = np.median(np.diff(t_centers)) if len(t_centers) > 1 else 1.0
                t_edges = np.concatenate(([t_centers[0] - dt / 2], t_centers + dt / 2))
                y_edges = np.arange(n_units + 1)

                gs = gridspec.GridSpecFromSubplotSpec(
                    2, 2,
                    subplot_spec=outer[0, ci],
                    width_ratios=[8, 1.8],
                    height_ratios=[1.2, 8],
                    wspace=0.05,
                    hspace=0.05,
                )

                ax_top = fig.add_subplot(gs[0, 0])
                ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_top)
                ax_right = fig.add_subplot(gs[1, 1], sharey=ax_heat)

                quad = ax_heat.pcolormesh(
                    t_edges, y_edges, M,
                    cmap="bwr", norm=norm, shading="flat", antialiased=False
                )
                mappables.append(quad)

                # Column sums
                col_sums = M.sum(axis=0)
                ax_top.plot(t_centers, col_sums, linewidth=1.0)
                ax_top.axvline(0.0, linestyle=":", linewidth=1)
                ax_top.set_xlim(time_window)
                ax_top.set_ylabel("Σ rows", fontsize=9)
                ax_top.set_xticks(np.linspace(time_window[0], time_window[1], 5))
                ax_top.set_xticklabels([f"{x:.2f}" for x in np.linspace(time_window[0], time_window[1], 5)], fontsize=8)


                # ---- heatmap y-ticks (make sure they exist & are visible) ----
                ax_heat.set_xlim(time_window)
                ax_heat.set_ylim(0, n_units)
                ax_heat.set_xlabel("Time from fixation onset (s)")
                ax_heat.axvline(0.0, linestyle=":", linewidth=1)

                # time ticks
                ax_heat.set_xticks(np.linspace(time_window[0], time_window[1], 5))
                ax_heat.set_xticklabels([f"{x:.2f}" for x in np.linspace(time_window[0], time_window[1], 5)], fontsize=9)

                # unit-number ticks at row centers
                if n_units > 0:
                    step = max(1, n_units // 10)
                    ytick_pos = np.arange(0.5, n_units + 0.5, step)
                    ytick_lab = [str(int(i + 0.5)) for i in np.arange(0, n_units, step)]
                    ax_heat.set_yticks(ytick_pos)
                    ax_heat.set_yticklabels(ytick_lab, fontsize=9)
                ax_heat.set_ylabel("Units (sorted by argmax ↓ early→late)")

                # ---- right marginal (row sums), perfectly aligned to heatmap rows ----
                row_sums = M.sum(axis=1)              # length n_units
                y_centers = np.arange(n_units) + 0.5  # row centers match pcolormesh cells

                # draw bars centered on row centers; height=1.0 to match cell height
                ax_right.barh(y_centers, row_sums, height=1.0, align="center")

                # lock y-limits to heatmap so bars and cells line up exactly
                ax_right.set_ylim(ax_heat.get_ylim())

                # keep x label
                ax_right.set_xlabel("Σ cols", fontsize=9)

                # hide only ax_right's tick artists (not the shared locator/formatter)
                for tick in ax_right.yaxis.get_major_ticks():
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)

                # hide y spines on the right panel only
                ax_right.spines["left"].set_visible(False)
                ax_right.spines["right"].set_visible(False)

                pretty_region = str(region) if region is not None else "UnknownRegion"
                ax_top.set_title(f"{pretty_region} (n={n_units}/{total_units} sig units)", pad=6)

            if mappables:
                cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
                cb = fig.colorbar(mappables[-1], cax=cax)
                cb.set_label("(I - N) / (I + N)")

            fig.subplots_adjust(left=0.06, right=0.90, top=0.92, bottom=0.12)

            for ext in export_formats:
                out_path = out_dir / f"{outfile_basename}.{ext}"
                if overwrite or not out_path.exists():
                    fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)



    def plot_region_violin_summaries(
        self,
        export_formats: tuple[str, ...] = ("pdf",),
        overwrite: bool = True,
        illustrator_friendly: bool = True,
        time_window: tuple[float, float] = (-0.5, 0.5),
        out_basename_prefix: str = "region_violin",
    ) -> None:
        """
        Create three separate figures (Illustrator-friendly vector PDFs):
          1) Violin of max-index location (time) per region + pairwise Welch t-tests
          2) Violin of index-sum per unit (Σ over time) per region + pairwise Welch t-tests
          3) Violin of min-index location (time) per region + pairwise Welch t-tests

        Only SIGNIFICANT units are included. The index is:
            (interactive - noninteractive) / (interactive + noninteractive)
        within `time_window`.
        """
        from itertools import combinations

        sig_df = self._load_or_compute_significance(alpha=0.05, test_window=time_window)
        if sig_df.empty:
            logger.warning("No significance results available; skipping violin plotting.")
            return

        sig_df = sig_df[sig_df["significant"] == True].copy()
        if sig_df.empty:
            logger.info("No significant units to plot in violins.")
            return

        # Extract per-unit metrics for each region
        regions = [r for r in sig_df["region"].dropna().unique().tolist()]
        if len(regions) == 0:
            regions = ["UnknownRegion"]

        per_region = {r: {"max_time": [], "min_time": [], "sum_index": []} for r in regions}

        for region in regions:
            rdf = sig_df[sig_df["region"] == region] if region != "UnknownRegion" else sig_df[sig_df["region"].isna()]
            for _, row in rdf.iterrows():
                t = np.asarray(row["time_axis"], dtype=float)
                m_int = np.asarray(row["mean_psth_interactive"], dtype=float)
                m_non = np.asarray(row["mean_psth_non_interactive"], dtype=float)
                if m_int is None or m_non is None:
                    continue
                mask = (t >= time_window[0]) & (t <= time_window[1])
                if not mask.any():
                    continue
                t_win = t[mask]
                x = m_int[mask]
                y = m_non[mask]

                denom = x + y
                idx = np.zeros_like(denom, dtype=float)
                nz = denom != 0
                idx[nz] = (x[nz] - y[nz]) / denom[nz]

                # metrics
                per_region[region]["max_time"].append(float(t_win[np.argmax(idx)]))
                per_region[region]["min_time"].append(float(t_win[np.argmin(idx)]))
                per_region[region]["sum_index"].append(float(idx.sum()))

        # Helper to make one violin figure + pairwise p-value table
        def _make_violin(metric_key: str, y_label: str, title_suffix: str, fname_suffix: str):
            # Assemble data
            cats = []
            data = []
            means = []
            for r in regions:
                vals = np.asarray(per_region[r][metric_key], dtype=float)
                cats.append(str(r))
                data.append(vals)
                means.append(vals.mean() if vals.size > 0 else np.nan)

            # Pairwise Welch t-tests across regions
            from scipy.stats import ttest_ind
            from itertools import combinations
            pvals = { (a, b): np.nan for a, b in combinations(range(len(regions)), 2) }
            for i, j in combinations(range(len(regions)), 2):
                xi, xj = data[i], data[j]
                if xi.size >= 2 and xj.size >= 2:
                    pvals[(i, j)] = ttest_ind(xi, xj, equal_var=False, nan_policy="omit").pvalue

            # Colors per region (Tab10)
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("tab10")
            region_colors = [cmap(i % 10) for i in range(len(regions))]

            # Figure
            ctx = rc_context(self._ILLUSTRATOR_RC) if illustrator_friendly else rc_context()
            with ctx:
                fig = plt.figure(figsize=(max(6, 1.8 * len(regions)) + 3.0, 4.8))
                # Main axis for violins
                ax = fig.add_axes([0.08, 0.15, 0.62, 0.76])

                # Violin plot (vector). We'll recolor bodies individually.
                parts = ax.violinplot(
                    data,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )

                # Color and outline each violin body
                for i, pc in enumerate(parts["bodies"]):
                    pc.set_facecolor(region_colors[i])
                    pc.set_edgecolor("black")
                    pc.set_linewidth(0.8)

                # Quartiles (Q1, Median, Q3) per group as horizontal ticks sized to violin width
                # Also overlay individual data points with light jitter
                rng = np.random.default_rng(0)  # deterministic jitter
                for i, vals in enumerate(data, start=1):
                    if vals.size == 0:
                        continue
                    q1, med, q3 = np.percentile(vals, [25, 50, 75])

                    # Determine violin half-width from polygon extents
                    body = parts["bodies"][i - 1]
                    verts = body.get_paths()[0].vertices
                    x_min, x_max = float(np.min(verts[:, 0])), float(np.max(verts[:, 0]))
                    half_w = (x_max - x_min) / 2.0
                    tick_w = min(half_w * 0.9, 0.3)  # keep neat width

                    # Draw quartile ticks (vector lines)
                    ax.hlines(q1, i - tick_w, i + tick_w, linewidth=1.0, colors="black")
                    ax.hlines(med, i - tick_w, i + tick_w, linewidth=1.4, colors="black")
                    ax.hlines(q3, i - tick_w, i + tick_w, linewidth=1.0, colors="black")

                    # Overlay data points (no alpha, subtle jitter)
                    jitter = rng.uniform(-min(0.08, half_w * 0.4), min(0.08, half_w * 0.4), size=vals.size)
                    ax.scatter(
                        np.full(vals.shape, i) + jitter,
                        vals,
                        s=10,              # marker size
                        marker="o",
                        edgecolors="black",
                        linewidths=0.4,
                        facecolors="white",  # high contrast over colored violin
                        zorder=3,
                    )

                # Overlay means as points + thin connecting line
                x_pos = np.arange(1, len(regions) + 1)
                ax.plot(x_pos, means, color="black", linewidth=1.0)
                ax.scatter(x_pos, means, color="black", s=20, zorder=4)

                # Cosmetics
                ax.set_xticks(x_pos)
                ax.set_xticklabels(cats, rotation=0)
                ax.set_ylabel(y_label)
                ax.set_title(f"{title_suffix}")

                # Legend mapping region -> color (vector patches)
                from matplotlib.patches import Patch
                legend_handles = [Patch(facecolor=region_colors[i], edgecolor="black", label=cats[i]) for i in range(len(cats))]
                ax.legend(handles=legend_handles, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

                # Pairwise p-value matrix as a side panel (vector text)
                ax_tbl = fig.add_axes([0.75, 0.15, 0.22, 0.76])
                ax_tbl.axis("off")
                lines = [f"Pairwise Welch t-tests (p-values)"]
                for i, j in combinations(range(len(regions)), 2):
                    rij = f"{cats[i]} vs {cats[j]}: "
                    pv = pvals[(i, j)]
                    lines.append(rij + ("n/a" if np.isnan(pv) else f"{pv:.3g}"))
                ax_tbl.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10)

                # Export
                base_plot_dir = Path(getattr(self.config, "plots_dir", getattr(self.config, "plot_dir", ".")))
                out_dir = base_plot_dir / "psth" / "interactive_units"
                out_dir.mkdir(parents=True, exist_ok=True)
                for ext in export_formats:
                    out_path = out_dir / f"{out_basename_prefix}_{fname_suffix}.{ext}"
                    if overwrite or not out_path.exists():
                        fig.savefig(out_path, bbox_inches="tight")
                plt.close(fig)


        # Make all three figures
        _make_violin(
            metric_key="max_time",
            y_label="Time of max index (s)",
            title_suffix="Max index location per region",
            fname_suffix="max_index_time",
        )
        _make_violin(
            metric_key="sum_index",
            y_label="Σ Index per unit",
            title_suffix="Index sum per unit per region",
            fname_suffix="index_sum",
        )
        _make_violin(
            metric_key="min_time",
            y_label="Time of min index (s)",
            title_suffix="Min index location per region",
            fname_suffix="min_index_time",
        )

