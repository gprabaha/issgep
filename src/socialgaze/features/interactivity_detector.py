# src/socialgaze/features/interactivity_detector.py
from __future__ import annotations

import os
import logging
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d

from socialgaze.utils.saving_utils import save_df_to_pkl
from socialgaze.utils.loading_utils import load_df_from_pkl
from socialgaze.config.interactivity_config import InteractivityConfig

logger = logging.getLogger(__name__)


class InteractivityDetector:
    def __init__(self, config):
        """
        Initializes the interactivity detector.

        Args:
            config (InteractivityConfig): Configuration object with file paths and parameters.
        """
        self.config = config
        self.fixation_type = config.fixation_type_to_process
        self.output_path = config.mutual_fixation_density_path
        self.num_cpus = config.num_cpus

        self.fix_binary_vector_df: Optional[pd.DataFrame] = None
        self.mutual_fixation_density: Optional[pd.DataFrame] = None
        self.interactivity_periods: Optional[pd.DataFrame] = None


    def detect_mutual_face_fix_density(self, overwrite: bool = False) -> None:
        """
        Computes or loads mutual fixation density and stores it in self.mutual_fixation_density.

        Args:
            overwrite (bool): If True, forces recomputation even if file exists.
        """
        if not overwrite and os.path.exists(self.output_path):
            logger.info(f"Mutual fixation density already exists. Loading from: {self.output_path}")
            self.mutual_fixation_density = self.load_fix_densities()
            return

        if self.fix_binary_vector_df is None:
            logger.info(f"Loading fixation binary vector DataFrame from: {self.config.fix_binary_vector_df_path}")
            self.fix_binary_vector_df = load_df_from_pkl(self.config.fix_binary_vector_df_path)

        logger.info(f"Computing mutual {self.fixation_type} fixation density...")

        session_groups = self.fix_binary_vector_df.groupby('session_name')

        if self.config.use_parallel:
            logger.info(f"Running in parallel using {self.num_cpus} CPUs")
            results = Parallel(n_jobs=self.num_cpus)(
                delayed(self._get_fix_density_in_session)(session_name, session_df)
                for session_name, session_df in tqdm(session_groups, desc="Processing Sessions")
            )
        else:
            logger.info("Running in serial mode")
            results = [
                self._get_fix_density_in_session(session_name, session_df)
                for session_name, session_df in tqdm(session_groups, desc="Processing Sessions")
            ]

        flattened_results = [entry for session_result in results for entry in session_result]
        self.mutual_fixation_density = pd.DataFrame(flattened_results)
        logger.info("Saving mutual fixation density...")
        self.save_fix_densities()



    def get_density(self, session_name: str = None, run_number: str = None) -> pd.DataFrame:
        """
        Returns the computed mutual fixation density, optionally filtered by session or run.
        If not already loaded, attempts to load from the saved pickle file.
        Args:
            session_name (str, optional): Filter by session name.
            run_number (str, optional): Filter by run number.
        Returns:
            pd.DataFrame: Filtered or full mutual fixation density.
        Raises:
            ValueError: If density is neither loaded nor found on disk.
        """
        if self.mutual_fixation_density is None:
            if os.path.exists(self.output_path):
                logger.info(f"Loading mutual fixation density from disk: {self.output_path}")
                self.mutual_fixation_density = self.load_fix_densities()
            else:
                raise ValueError(
                    f"Mutual fixation density not available in memory or on disk. "
                    f"Call `.run()` to compute it."
                )
        df = self.mutual_fixation_density
        pdb.set_trace()
        if session_name:
            df = df[df['session_name'] == session_name]
        if run_number:
            df = df[df['run_number'] == run_number]
        return df.reset_index(drop=True)


    def save_fix_densities(self) -> None:
        """
        Saves the mutual fixation density dataframe to the configured output path.
        The dataframe must be present in `self.mutual_fixation_density`. Raises a warning if it is None or empty.
        """
        if self.mutual_fixation_density is None or self.mutual_fixation_density.empty:
            logger.warning("Mutual fixation density is empty or None. Nothing to save.")
            return
        save_df_to_pkl(self.mutual_fixation_density, self.output_path)
        logger.info(f"Mutual fixation density saved to {self.output_path}")


    def load_fix_densities(self) -> pd.DataFrame:
        """
        Loads the mutual fixation density dataframe from the configured output path.
        Returns:
            pd.DataFrame: The loaded mutual fixation density dataframe.
        Raises:
            FileNotFoundError: If the output file does not exist.
        """
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"No fixation density file found at: {self.output_path}")
        
        self.mutual_fixation_density = load_df_from_pkl(self.output_path)
        logger.info(f"Mutual fixation density loaded from {self.output_path}")
        return self.mutual_fixation_density


    def compute_interactivity_periods(self, overwrite: bool = False) -> None:
        """
        Computes or loads periods of interactivity (start/stop indices) for each session and run,
        using thresholded mutual fixation density. Stores the result in self.interactivity_df.
        Args:
            overwrite (bool): If True, recompute and overwrite existing file. Otherwise, try to load.
        """
        output_path = self.config.interactivity_df_path

        if not overwrite and os.path.exists(output_path):
            logger.info(f"Interactivity periods already exist. Loading from: {output_path}")
            self.interactivity_periods = load_df_from_pkl(output_path)
            return
        if self.mutual_fixation_density is None:
            logger.info("No mutual fixation density in memory, attempting to load.")
            self.mutual_fixation_density = self.load_fix_densities()
        results = []
        for (session, run), group in self.mutual_fixation_density.groupby(["session_name", "run_number"]):
            mutual = np.array(group["mutual_density"].values[0])
            threshold = self.config.interactivity_threshold * np.mean(mutual)
            is_interactive = mutual > threshold
            periods = _get_interactive_periods(is_interactive)
            for start, stop in periods:
                results.append({
                    "session_name": session,
                    "run_number": run,
                    "start": start,
                    "stop": stop
                })
        self.interactivity_periods = pd.DataFrame(results)
        return self.interactivity_periods 


    def save_interactivity_periods(self, path: str = None):
        """Saves the interactivity_df to disk as a pickle file."""
        if self.interactivity_periods is None:
            raise ValueError("No interactivity data to save.")
        path = path or self.config.interactivity_df_path
        save_df_to_pkl(self.interactivity_periods, path)
        logger.info("Saved interactivity dataframe to %s", path)


    def load_interactivity_periods(self, path: str = None) -> pd.DataFrame:
        """Loads the interactivity_df from disk."""
        path = path or self.config.interactivity_df_path
        self.interactivity_periods = load_df_from_pkl(path)
        logger.info("Loaded interactivity dataframe from %s", path)
        return self.interactivity_periods

    def get_interactivity_periods(self, session_name: str = None, run_number: str = None) -> pd.DataFrame:
        """
        Returns the interactivity periods, optionally filtered by session and run.
        Loads from disk if not already in memory.
        """
        if self.interactivity_periods is None:
            logger.info("Interactivity dataframe not in memory. Loading from disk.")
            self.load_interactivity_periods()
        df = self.interactivity_periods
        if session_name:
            df = df[df["session_name"] == session_name]
        if run_number:
            df = df[df["run_number"] == run_number]
        return df.reset_index(drop=True)


    def _get_fix_density_in_session(self, session_name, session_df):
        """
        Computes mutual fixation density for each run in a session.

        For each run, this function retrieves binary fixation vectors for both agents (m1 and m2),
        calculates their fixation durations and inter-fixation intervals (IFIs), and smooths the
        vectors using Gaussian filters whose standard deviation is based on the mean of fixation 
        duration and IFI. The mutual fixation density is then computed as the geometric mean of
        the normalized densities from both agents.

        Args:
            session_name (str): Name of the session being processed.
            session_df (pd.DataFrame): Dataframe containing fixation binary vectors for the session.

        Returns:
            List[Dict]: A list of dictionaries, one per run, containing session metadata,
                        fixation metrics, and density arrays for m1, m2, and mutual fixation.
        """

        run_groups = session_df.groupby("run_number")
        results = []
        for run_number, run_df in run_groups:
            m1 = run_df[(run_df.agent == "m1") & (run_df.fixation_type == self.fixation_type)]
            m2 = run_df[(run_df.agent == "m2") & (run_df.fixation_type == self.fixation_type)]
            if m1.empty or m2.empty:
                continue

            m1_vec = np.array(m1.binary_vector.values[0])
            m2_vec = np.array(m2.binary_vector.values[0])

            m1_fix_dur, m1_ifi = compute_fixation_metrics(m1_vec)
            m2_fix_dur, m2_ifi = compute_fixation_metrics(m2_vec)
            m1_sigma = (m1_fix_dur + m1_ifi) / 2
            m2_sigma = (m2_fix_dur + m2_ifi) / 2

            min_len = min(len(m1_vec), len(m2_vec))
            m1_vec = m1_vec[:min_len]
            m2_vec = m2_vec[:min_len]

            m1_density = gaussian_filter1d(m1_vec.astype(float), sigma=m1_sigma, mode='constant')
            m2_density = gaussian_filter1d(m2_vec.astype(float), sigma=m2_sigma, mode='constant')

            m1_density_norm = normalize_density(m1_density)
            m2_density_norm = normalize_density(m2_density)
            mutual = np.sqrt(m1_density_norm * m2_density_norm)
            mutual_norm = normalize_density(mutual)

            results.append({
                'session_name': session_name,
                'run_number': run_number,
                'fixation_type': self.fixation_type,
                'm1_fix_dur': m1_fix_dur,
                'm2_fix_dur': m2_fix_dur,
                'm1_ifi': m1_ifi,
                'm2_ifi': m2_ifi,
                'm1_sigma': m1_sigma,
                'm2_sigma': m2_sigma,
                'm1_density': list(m1_density_norm),
                'm2_density': list(m2_density_norm),
                'mutual_density': list(mutual_norm)
            })

        return results


def compute_fixation_metrics(binary_vector):
    """
    Computes average fixation duration and inter-fixation interval (IFI) from a binary vector.

    Args:
        binary_vector (np.ndarray or List[int]): A binary vector where 1 indicates a fixation
                                                 and 0 indicates absence of fixation.

    Returns:
        Tuple[float, float]: Mean fixation duration and mean inter-fixation interval.
                             Returns (0, 0) if no fixation is detected.
    """
    vec = np.array(binary_vector)
    if np.all(vec == 0):
        return 0, 0
    if np.all(vec == 1):
        return len(vec), 0
    changes = np.where(np.diff(np.pad(vec, (1, 1), 'constant')) != 0)[0]
    durations = np.diff(changes)
    fix_durations = durations[::2]
    ifi_durations = durations[1::2] if len(durations) > 1 else [0]
    return np.mean(fix_durations), np.mean(ifi_durations)


def normalize_density(arr):
    """
    Normalizes an array to the range [0, 1].
    Args:
        arr (np.ndarray): Input array.
    Returns:
        np.ndarray: Normalized array where the minimum becomes 0 and maximum becomes 1.
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)


def _get_interactive_periods(binary_vector):
    """
    Returns a list of (start, stop) indices where stop is inclusive.
    Safe for use with: v[start:stop+1]
    """
    vec = np.asarray(binary_vector, dtype=int)
    if np.all(vec == 0):
        return []

    padded = np.pad(vec, (1, 1), constant_values=0)
    change_indices = np.flatnonzero(np.diff(padded))

    starts = change_indices[::2]
    stops = change_indices[1::2] - 1  # make 'stop' inclusive

    return list(zip(starts, stops))



from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Literal

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pdb


class InteractivityPlotter(InteractivityDetector):
    """
    Plotting module for interactivity-related visualizations.
    Leverages precomputed face–face density & interactivity periods when available.
    """

    # ======== Public API ========

    def preview_random_runs(
        self,
        fixation_detector,
        pairing: Tuple[str, str, str, str],  # (agentA, catA, agentB, catB)
        n_samples: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        df = fixation_detector.get_behavior_data("fixations")
        agentA, catA, agentB, catB = pairing

        cand = self._runs_with_both_categories(df, agentA, catA, agentB, catB)
        if cand.empty:
            print("No runs satisfy the pairing criteria.")
            return

        picks = cand.sample(n=min(n_samples, len(cand)), random_state=seed)
        for session, run in picks.itertuples(index=False, name=None):
            self._plot_run_panel(
                df=df, session=session, run=run,
                pairing=pairing,
                export_path=None,
                illustrator_friendly=False
            )

    def export_representative_runs(
        self,
        fixation_detector,
        pairing: Tuple[str, str, str, str],
        session_runs: Sequence[Tuple[str, int]],
        export_dir: Path,
        dpi: int = 300,
    ) -> None:
        export_dir.mkdir(parents=True, exist_ok=True)
        df = fixation_detector.get_behavior_data("fixations")
        for session, run in session_runs:
            out = export_dir / f"{session}_{run}.pdf"
            self._plot_run_panel(
                df=df, session=session, run=run,
                pairing=pairing,
                export_path=out,
                illustrator_friendly=True,
                dpi=dpi
            )

    def plot_face_fixation_pies_by_agent(
        self,
        fixation_detector,
        export_dir: Optional[Path] = None,
    ) -> None:
        """
        Per agent:
          Pie A (all face fixations): Non-interactive vs Interactive (explode Interactive)
          Pie B (interactive only):   Non-mutual vs Mutual           (explode Mutual)
        Uses precomputed face–face interactivity windows.
        """
        df = fixation_detector.get_behavior_data("fixations")
        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)

        # Precomputed interactivity windows (face–face)
        ff_windows = self._load_all_interactive_windows()

        for agent in ("m1", "m2"):
            counts_all = self._count_interactive_vs_noninteractive_faces(
                df=df, agent=agent, interactive_windows=ff_windows
            )
            counts_inter = self._count_mutual_vs_nonmutual_within_interactive(
                df=df, agent=agent, interactive_windows=ff_windows
            )

            # Pie A
            labels_A = ["Non-interactive", "Interactive"]
            values_A = [counts_all["non_interactive"], counts_all["interactive"]]
            total_A = sum(values_A) or 1
            explode_A = (0.0, 0.08)

            figA, axA = plt.subplots(figsize=(5.2, 5.2))
            wedges_A, _, _ = axA.pie(
                values_A, startangle=90, explode=explode_A,
                autopct=self._autopct(values_A), pctdistance=0.75,
                wedgeprops=dict(linewidth=1, edgecolor="white")
            )
            axA.axis("equal")
            axA.set_title(f"Face fixations — {agent}\nInteractive vs Non-interactive")
            axA.legend(
                wedges_A,
                [self._label_count_pct(lbl, val, total_A) for lbl, val in zip(labels_A, values_A)],
                loc="center left", bbox_to_anchor=(1, 0.5)
            )
            if export_dir is not None:
                self._set_illustrator_pdf_rc()
                figA.savefig(export_dir / f"facefix_pieA_interactive_{agent}.pdf", bbox_inches="tight")
                plt.close(figA)
            else:
                plt.show()

            # Pie B
            labels_B = ["Non-mutual (interactive)", "Mutual (interactive)"]
            values_B = [counts_inter["interactive_non_mutual"], counts_inter["interactive_mutual"]]
            total_B = sum(values_B) or 1
            explode_B = (0.0, 0.08)

            figB, axB = plt.subplots(figsize=(5.2, 5.2))
            wedges_B, _, _ = axB.pie(
                values_B, startangle=90, explode=explode_B,
                autopct=self._autopct(values_B), pctdistance=0.75,
                wedgeprops=dict(linewidth=1, edgecolor="white")
            )
            axB.axis("equal")
            axB.set_title(f"Face fixations — {agent}\nMutual fraction (within interactive)")
            axB.legend(
                wedges_B,
                [self._label_count_pct(lbl, val, total_B) for lbl, val in zip(labels_B, values_B)],
                loc="center left", bbox_to_anchor=(1, 0.5)
            )
            if export_dir is not None:
                self._set_illustrator_pdf_rc()
                figB.savefig(export_dir / f"facefix_pieB_mutual_{agent}.pdf", bbox_inches="tight")
                plt.close(figB)
            else:
                plt.show()

    # ======== Core per-run plot (uses precomp for face–face) ========

    def _plot_run_panel(
        self,
        df: pd.DataFrame,
        session: str,
        run: int,
        pairing: tuple[str, str, str, str],
        export_path: Optional[Path],
        illustrator_friendly: bool = False,
        dpi: int = 300,
    ) -> None:
        agentA, catA, agentB, catB = pairing

        # Intervals for broken bars
        A = self._intervals_for(df, session, str(run), agentA, catA)
        B = self._intervals_for(df, session, str(run), agentB, catB)
        use_precomp = (agentA, catA, agentB, catB) == ("m1", "face", "m2", "face")
        density = None
        interactive_spans: list[tuple[int, int]] = []

        if use_precomp:
            dens = self._get_precomputed_density(session, run)
            if dens is not None:
                density = dens
            interactive_spans = self._get_interactivity_spans(session, run)

        # If not precomputed (or missing), compute on the fly
        if density is None:
            L = 0
            if len(A):
                L = max(L, int(np.max([e for _, e in A])))
            if len(B):
                L = max(L, int(np.max([e for _, e in B])))
            if L == 0:
                return

            a = np.zeros(L, dtype=np.float32)
            b = np.zeros(L, dtype=np.float32)
            for s, e in A:
                a[s:e] = 1.0
            for s, e in B:
                b[s:e] = 1.0

            sigma = self._choose_sigma_samples(A, B)
            a_s = gaussian_filter1d(a, sigma=sigma, mode="nearest") if sigma and sigma > 0 else a
            b_s = gaussian_filter1d(b, sigma=sigma, mode="nearest") if sigma and sigma > 0 else b

            # Normalize individual densities
            m1_density_norm = normalize_density(a_s)
            m2_density_norm = normalize_density(b_s)

            # Mutual density = sqrt(product), then normalized
            mutual = np.sqrt(m1_density_norm * m2_density_norm)
            mut = normalize_density(mutual)
        else:
            mut = np.asarray(density["mutual_density"], dtype=float)

        thr = float(self.config.interactivity_threshold) * float(mut.mean() if mut.size else 0.0)
        if not interactive_spans:
            interactive_spans = self._mask_to_spans(mut >= thr)

        # --- Plot ---
        if illustrator_friendly:
            self._set_illustrator_pdf_rc()

        # Avoid constrained_layout to reduce clip paths
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 4.4), sharex=True)
        ax_int, ax_den = axes

        # Row 1: broken bars (no clipping)
        y0, height = 0.0, 1.0
        A_bars = [(s/1000.0, (e - s)/1000.0) for s, e in A]
        B_bars = [(s/1000.0, (e - s)/1000.0) for s, e in B]
        ax_int.broken_barh(
            A_bars, (y0, height),
            facecolors="C0", alpha=0.85,
            label=f"{agentA} {catA}",
            clip_on=False, zorder=2
        )
        ax_int.broken_barh(
            B_bars, (y0, height),
            facecolors="C1", alpha=0.55,
            label=f"{agentB} {catB}",
            clip_on=False, zorder=1
        )
        ax_int.set_ylim(-0.1, 1.1)
        ax_int.set_yticks([])
        ax_int.set_ylabel("Fixations")
        ax_int.legend(loc="upper right", frameon=False)
        ax_int.set_title(f"{session}, run {run} — {agentA}:{catA} vs {agentB}:{catB}")

        # Row 2: shaded spans (behind) + mutual density (chunked) + threshold
        t = np.arange(mut.size) / 1000.0

        # Opaque light color to avoid AI opacity masks
        span_color = "#e7f0fb"

        # Spans behind everything; no clipping
        for s, e in interactive_spans:
            ax_den.axvspan(
                s/1000.0, e/1000.0,
                facecolor=span_color,
                clip_on=False, zorder=0
            )

        # Mutual density, preserved fully but split into subpaths
        _plot_chunked_line(
            ax_den, t, mut,
            max_points=10000,
            label="mutual density",
            linewidth=2,
            zorder=3,
            clip_on=False,
            color="C0"   # or any consistent hex/RGB
        )


        # Threshold line above spans; no clipping
        ax_den.axhline(
            thr, linestyle="--", linewidth=1.6,
            label=f"θ·mean (θ={self.config.interactivity_threshold:g})",
            clip_on=False, zorder=2
        )

        ax_den.set_xlabel("Time (s)")
        ax_den.set_ylabel("Density")
        ax_den.legend(loc="upper right", frameon=False)

        # Final layout AFTER creating all artists
        fig.tight_layout()

        if export_path is None:
            plt.show()
        else:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(export_path, bbox_inches="tight")  # vector; dpi irrelevant for vector
            plt.close(fig)


    # ======== Windows & counts (use precomputed face–face periods) ========

    def _load_all_interactive_windows(self) -> dict[Tuple[str, int], List[Tuple[int, int]]]:
        """
        Build {(session, run): [(start, stop), ...]} from precomputed interactivity periods.
        """
        df = self.get_interactivity_periods()  # face–face only
        out: dict[Tuple[str, int], List[Tuple[int, int]]] = {}
        if df is None or df.empty:
            return out
        for (session, run), g in df.groupby(["session_name", "run_number"]):
            spans = [(int(s), int(e)) for s, e in zip(g["start"].to_numpy(), g["stop"].to_numpy())]
            if spans:
                out[(session, run)] = spans
        return out

    def _count_interactive_vs_noninteractive_faces(
        self,
        df: pd.DataFrame,
        agent: Literal["m1", "m2"],
        interactive_windows: dict[Tuple[str, int], List[Tuple[int, int]]],
    ) -> dict[str, int]:
        counts = {"interactive": 0, "non_interactive": 0}
        for (session, run), _ in df.groupby(["session_name", "run_number"]):
            F = self._intervals_for(df, session, run, agent, "face")
            if not F:
                continue
            IW = interactive_windows.get((session, run), [])
            for s, e in F:
                if self._interval_overlaps_any((s, e), IW):
                    counts["interactive"] += 1
                else:
                    counts["non_interactive"] += 1
        return counts

    def _count_mutual_vs_nonmutual_within_interactive(
        self,
        df: pd.DataFrame,
        agent: Literal["m1", "m2"],
        interactive_windows: dict[Tuple[str, int], List[Tuple[int, int]]],
    ) -> dict[str, int]:
        other = "m2" if agent == "m1" else "m1"
        counts = {"interactive_mutual": 0, "interactive_non_mutual": 0}

        for (session, run), _ in df.groupby(["session_name", "run_number"]):
            F = self._intervals_for(df, session, run, agent, "face")
            if not F:
                continue
            O = self._intervals_for(df, session, run, other, "face")
            IW = interactive_windows.get((session, run), [])

            for s, e in F:
                if not self._interval_overlaps_any((s, e), IW):
                    continue  # skip non-interactive
                if self._interval_overlaps_any((s, e), O):
                    counts["interactive_mutual"] += 1
                else:
                    counts["interactive_non_mutual"] += 1
        return counts

    # ======== Low-level utils ========

    def _runs_with_both_categories(
        self, df: pd.DataFrame, a: str, ca: str, b: str, cb: str
    ) -> pd.DataFrame:
        dA = df.query("agent == @a and category == @ca")
        dB = df.query("agent == @b and category == @cb")
        left = dA.groupby(["session_name", "run_number"]).size().reset_index(name="nA")
        right = dB.groupby(["session_name", "run_number"]).size().reset_index(name="nB")
        cand = left.merge(right, on=["session_name", "run_number"], how="inner")
        return cand[["session_name", "run_number"]]

    def _intervals_for(
        self, df: pd.DataFrame, session: str, run: str, agent: str, category: str
    ) -> List[Tuple[int, int]]:
        rows = df.query(
            "session_name == @session and run_number == @run and agent == @agent and category == @category"
        )[["start", "stop"]]
        if rows.empty:
            return []
        return [tuple(map(int, x)) for x in rows.to_numpy()]

    def _interval_overlaps_any(self, iv: Tuple[int, int], spans: List[Tuple[int, int]]) -> bool:
        s, e = iv
        for ss, ee in spans:
            if s < ee and ss < e:
                return True
        return False

    def _mask_to_spans(self, m: np.ndarray) -> List[Tuple[int, int]]:
        if m.size == 0:
            return []
        idx = np.flatnonzero(m.astype(np.int8))
        if idx.size == 0:
            return []
        gaps = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[gaps + 1]]
        stops = np.r_[idx[gaps], idx[-1]]
        return [(int(s), int(e) + 1) for s, e in zip(starts, stops)]

    def _choose_sigma_samples(self, A: List[Tuple[int, int]], B: List[Tuple[int, int]]) -> int:
        """
        Sigma (in samples) = mean( mean_fix_dur_across_agents , mean_IFI_across_agents )

        - mean_fix_dur(X): average of (stop - start) for intervals in X
        - mean_IFI(X): average gap between consecutive intervals in X (sorted by start);
                    gaps are max(0, next.start - prev.stop) to ignore overlap negatives.

        Falls back gracefully if one side (A or B) lacks intervals.
        Returns 0 if nothing usable.
        """
        import numpy as np

        def _mean_fix_duration(X):
            if not X:
                return np.nan
            return float(np.mean([e - s for s, e in X]))

        def _mean_ifi(X):
            if not X or len(X) < 2:
                return np.nan
            Xs = sorted(X, key=lambda t: t[0])
            gaps = [max(0, Xs[i+1][0] - Xs[i][1]) for i in range(len(Xs) - 1)]
            if not gaps:
                return np.nan
            return float(np.mean(gaps))

        m_fix_A, m_fix_B = _mean_fix_duration(A), _mean_fix_duration(B)
        m_ifi_A, m_ifi_B = _mean_ifi(A), _mean_ifi(B)

        mean_fix = np.nanmean([m_fix_A, m_fix_B])  # across agents
        mean_ifi = np.nanmean([m_ifi_A, m_ifi_B])  # across agents

        sigma = np.nanmean([mean_fix, mean_ifi])   # combine duration & IFI

        if np.isnan(sigma) or sigma <= 0:
            return 0
        return int(round(sigma))


    def _set_illustrator_pdf_rc(self) -> None:
        mpl.rcParams.update({
            "pdf.fonttype": 42,          # embed TrueType
            "ps.fonttype": 42,
            "path.simplify": False,      # keep exact paths (no weird merges)
            "pdf.compression": 0,        # avoid compression-related AI quirks
            "figure.dpi": 200,           # irrelevant for vector, fine to leave
        })


    def _autopct(self, values: List[int]):
        total = float(sum(values) or 1)
        def f(pct): return f"{pct:.1f}%"
        return f

    def _label_count_pct(self, label: str, val: int, total: int) -> str:
        pct = (100.0 * val / (total or 1))
        return f"{label}: {val} ({pct:.1f}%)"

    # ======== Precomputed accessors ========

    def _get_precomputed_density(self, session: str, run: int) -> Optional[dict]:
        """
        Returns dict with 'mutual_density' (np.array) if available for (session, run).
        Uses self.get_density(session, run) which loads from disk if needed.
        """
        try:
            dens_df = self.get_density(session_name=session, run_number=run)            
        except Exception:
            return None
        if dens_df is None or dens_df.empty:
            return None
        row = dens_df.iloc[0]
        return {
            "mutual_density": np.asarray(row["mutual_density"], dtype=float),
            "m1_density": np.asarray(row["m1_density"], dtype=float) if "m1_density" in row else None,
            "m2_density": np.asarray(row["m2_density"], dtype=float) if "m2_density" in row else None,
        }

    def _get_interactivity_spans(self, session: str, run: int) -> List[Tuple[int, int]]:
        """
        Returns precomputed [(start, stop), ...] for (session, run) from self.get_interactivity_periods().
        """
        try:
            df = self.get_interactivity_periods(session_name=session, run_number=run)
        except Exception:
            return []
        if df is None or df.empty:
            return []
        return [(int(s), int(e)) for s, e in zip(df["start"].to_numpy(), df["stop"].to_numpy())]
    

# --- Helper: plot long polylines as smaller subpaths (Illustrator-safe) ---
def _plot_chunked_line(
    ax, x, y, *, max_points: int = 10000, label: str | None = None,
    clip_on: bool = False, color=None, **kwargs
):
    """
    Plot a long x/y polyline as multiple subpaths of <= max_points each.
    - Preserves every vertex (no decimation).
    - Breaks at NaNs before chunking.
    - Only the first subpath receives the legend label.
    - All chunks use the same color.
    Returns a list of Line2D handles.
    """
    import numpy as np

    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape

    isn = np.isnan(x) | np.isnan(y)
    segments: list[tuple[int, int]] = []
    start = None
    for i, bad in enumerate(isn):
        if not bad and start is None:
            start = i
        if (bad or i == len(x) - 1) and start is not None:
            end = i if bad else i + 1  # stop-exclusive
            if end > start:
                segments.append((start, end))
            start = None

    handles = []
    first = True
    for s, e in segments:
        for cs in range(s, e, max_points):
            ce = min(cs + max_points, e)
            (h,) = ax.plot(
                x[cs:ce], y[cs:ce],
                label=(label if first and label else None),
                clip_on=clip_on,
                color=color,           # force same color
                **kwargs,
            )
            handles.append(h)
            first = False
    return handles

