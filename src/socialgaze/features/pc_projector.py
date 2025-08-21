# src/socialgaze/features/pc_projector.py

import pdb
import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

logger = logging.getLogger(__name__)


class PCProjector:
    def __init__(self, psth_extractor, pca_config):
        self.psth_extractor = psth_extractor
        self.pca_config = pca_config


    def run_face_obj_analysis(self):
        df = self.psth_extractor.fetch_psth("face_obj")
        comparison = "face_obj"
        for region, region_df in df.groupby("region"):
            logger.info(f"Doing PCA for region {region}: face vs object")
            X_face = self._build_matrix(region_df, "face")
            X_object = self._build_matrix(region_df, "object")
            self._run_pca_analysis(X_face, X_object, region, "face", "object", comparison)


    def run_int_nonint_face_analysis(self):
        df = self.psth_extractor.fetch_psth("int_non_int_face")
        comparison = "int_nonint_face"
        for region, region_df in df.groupby("region"):
            logger.info(f"Doing PCA for region {region}: face_interactive vs face_non_interactive")
            X_int = self._build_matrix(region_df, "face_interactive")
            X_nonint = self._build_matrix(region_df, "face_non_interactive")
            self._run_pca_analysis(X_int, X_nonint, region, "face_interactive", "face_non_interactive", comparison)


    def _build_matrix(self, df, category) -> np.ndarray:
        mat = np.stack(df[df["category"] == category]["avg_firing_rate"].apply(np.array))
        return mat.T


    def _run_pca_analysis(self, X1, X2, region, label1, label2, comparison):
        n_components = self.pca_config.n_components

        pca1 = PCA(n_components=n_components).fit(X1)
        evr_X1_on_X1 = pca1.explained_variance_ratio_
        evr_X1_on_X2 = self._explained_var_on_data(pca1, X2)

        pca2 = PCA(n_components=n_components).fit(X2)
        evr_X2_on_X2 = pca2.explained_variance_ratio_
        evr_X2_on_X1 = self._explained_var_on_data(pca2, X1)

        X_comb = np.concatenate([X1, X2], axis=0)
        # pdb.set_trace()
        pca_comb = PCA(n_components=n_components).fit(X_comb)
        evr_comb_X1_X2 = self._explained_var_on_data(pca_comb, X_comb) # pca_comb.explained_variance_ratio_
        evr_comb_X1 = self._explained_var_on_data(pca_comb, X1)
        evr_comb_X2 = self._explained_var_on_data(pca_comb, X2)

        self._plot_evr_bar(
            region, evr_X1_on_X1, evr_X1_on_X2, evr_X2_on_X2, evr_X2_on_X1,
            evr_comb_X1_X2, evr_comb_X1, evr_comb_X2,
            label1, label2, comparison
        )

        proj_X1 = pca_comb.transform(X1)
        proj_X2 = pca_comb.transform(X2)
        self._plot_trajectories_with_metrics(proj_X1, proj_X2, label1, label2, region, comparison)


    def _explained_var_on_data(self, pca: PCA, X: np.ndarray) -> np.ndarray:
        X_centered = X - pca.mean_
        total_var = np.sum(np.var(X_centered, axis=0))
        X_proj = pca.transform(X)
        var_proj = np.var(X_proj, axis=0)
        return var_proj / total_var



    def _plot_evr_bar(
        self, region, evr_X1_X1, evr_X1_X2, evr_X2_X2, evr_X2_X1,
        evr_comb_X1_X2, evr_comb_X1, evr_comb_X2,
        label1, label2, comparison
    ):
        pcs = np.arange(1, len(evr_X1_X1) + 1)

        fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

        # Subplot 1: Fit label1
        df1 = pd.DataFrame({
            f"{label1} EVR": evr_X1_X1,
            f"{label2} EVR": evr_X1_X2
        }, index=pcs)
        df1.plot(kind="bar", ax=axs[0])
        axs[0].set_title(f"Fit {label1}")
        axs[0].set_xlabel("PC")
        axs[0].set_ylabel("Explained Variance Ratio")

        # Subplot 2: Fit label2
        df2 = pd.DataFrame({
            f"{label1} EVR": evr_X2_X1,
            f"{label2} EVR": evr_X2_X2
        }, index=pcs)
        df2.plot(kind="bar", ax=axs[1])
        axs[1].set_title(f"Fit {label2}")
        axs[1].set_xlabel("PC")

        # Subplot 3: Fit both
        df3 = pd.DataFrame({
            f"{label1} EVR": evr_comb_X1,
            f"{label2} EVR": evr_comb_X2,
            f"{label1}+{label2} EVR": evr_comb_X1_X2
        }, index=pcs)
        df3.plot(kind="bar", ax=axs[2])
        axs[2].set_title("Fit Both")
        axs[2].set_xlabel("PC")

        fig.suptitle(f"{region} — {comparison} — Explained Variance per PC")
        plt.tight_layout()

        out_path = self.pca_config.evr_bars_dir / f"{comparison}__{region}.png"
        plt.savefig(out_path)
        logger.info(f"Saved EVR bar plot: {out_path}")
        plt.close()


    def _plot_trajectories_with_metrics(self, proj_X1, proj_X2, label1, label2, region, comparison):
        config = self.psth_extractor.config
        bin_size = config.psth_bin_size
        start, end = config.psth_window
        num_bins = proj_X1.shape[0]
        timeline = np.linspace(
            start + bin_size / 2,
            end - bin_size / 2,
            num_bins
        )

        # === Optional mean centering ===
        if self.pca_config.mean_center_for_angle:
            logger.info(f"Mean-centering PC projections for angle calculation: {region} — {comparison}")
            proj_X1_centered = proj_X1 - proj_X1.mean(axis=0, keepdims=True)
            proj_X2_centered = proj_X2 - proj_X2.mean(axis=0, keepdims=True)
        else:
            proj_X1_centered = proj_X1
            proj_X2_centered = proj_X2

        dist = np.linalg.norm(proj_X1 - proj_X2, axis=1)

        dot = np.sum(proj_X1_centered * proj_X2_centered, axis=1)
        norm1 = np.linalg.norm(proj_X1_centered, axis=1)
        norm2 = np.linalg.norm(proj_X2_centered, axis=1)
        angle_deg = np.degrees(np.arccos(np.clip(dot / (norm1 * norm2 + 1e-10), -1.0, 1.0)))

        fig = plt.figure(figsize=(16, 6))

        # === 3D PC trajectory on the left ===
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot(proj_X1[:, 0], proj_X1[:, 1], proj_X1[:, 2], label=label1, color='blue')
        ax1.plot(proj_X2[:, 0], proj_X2[:, 1], proj_X2[:, 2], label=label2, color='red')
        ax1.set_title(f"{region} — {comparison} — PC1–PC2–PC3")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.set_zlabel("PC3")
        ax1.legend()

        # === Euclidean Distance (top right) ===
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(timeline, dist, label="Euclidean Distance", color='black')
        ax2.set_title(f"{region} — {comparison} — Euclidean Distance")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Distance")
        ax2.legend()

        # === Angle (bottom right) ===
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.plot(timeline, angle_deg, label="Angle (deg)", color='green')
        ax3.set_title(f"{region} — {comparison} — Angle")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angle (deg)")
        ax3.legend()

        plt.tight_layout()

        out_path = self.pca_config.trajectories_dir / f"{comparison}__{region}.png"
        plt.savefig(out_path)
        logger.info(f"Saved 3D trajectory + metrics plot: {out_path}")
        plt.close()











# ----------------------------------------------------------------------
# Illustrator-friendly plotting subclass
# ----------------------------------------------------------------------
import json
from pathlib import Path
from itertools import combinations
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA


class PCProjectorPlotter(PCProjector):
    """
    Adds interactive 3D view selection, row-wise PC time-series plots, and
    violin plots of distance/angle with pairwise comparisons, while keeping
    figures Illustrator-friendly (live text, individually selectable paths).
    """

    # ---- rcParams tuned for vector-friendly export ----
    _ILLUSTRATOR_RCPARAMS = {
        # Keep fonts as text (Type 42) in PDF/PS; don't outline glyphs in SVG
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Avoid aggressive path simplification/clipping that can create big boxes
        "path.simplify": False,
        "savefig.transparent": True,
        # No global alpha that might rasterize
        "figure.dpi": 100,
    }

    def __init__(self, psth_extractor, pca_config):
        super().__init__(psth_extractor, pca_config)
        # Where we cache (elev, azim) per (comparison, region)
        self.view_cache_path: Path = getattr(
            self.pca_config, "view_cache_path",
            (self.pca_config.trajectories_dir / "view_cache.json")
        )
        self._ensure_dirs()

    # -------------------- Public API --------------------

    def pick_3d_views_for(self, comparison: str):
        """
        Show 3D PC1-3 trajectories per region for `comparison`.
        Rotate with mouse; press 'S' to save (elev, azim) for that region in cache.
        Close figure to proceed to the next region.
        Valid `comparison`: "face_obj" or "int_nonint_face".
        """
        self._apply_illustrator_rc()

        labels = self._labels_for_comparison(comparison)  # (label1, label2)
        df = self.psth_extractor.fetch_psth(comparison)

        cache = self._load_view_cache()
        cache.setdefault(comparison, {})

        for region, region_df in df.groupby("region"):
            proj_X1, proj_X2 = self._compute_combined_proj(region_df, *labels)
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection="3d")

            # Trajectories (vector paths)
            ax.plot(proj_X1[:, 0], proj_X1[:, 1], proj_X1[:, 2], label=labels[0])
            ax.plot(proj_X2[:, 0], proj_X2[:, 1], proj_X2[:, 2], label=labels[1])

            ax.set_title(f"{region} — {comparison} — Rotate, then press 'S' to save view")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
            ax.legend()

            # Preload cached view if available
            elev_az = cache[comparison].get(str(region))
            if elev_az is not None:
                ax.view_init(elev=elev_az["elev"], azim=elev_az["azim"])

            def _on_key(event):
                if event.key and event.key.lower() == "s":
                    elev, azim = float(ax.elev), float(ax.azim)
                    cache[comparison][str(region)] = {"elev": elev, "azim": azim}
                    self._save_view_cache(cache)
                    logger.info(f"Saved view for {comparison}/{region}: elev={elev:.2f}, azim={azim:.2f}")
                    # Display on-plot for reproducibility
                    ax.text2D(0.02, 0.98, f"Saved: elev={elev:.1f}, azim={azim:.1f}",
                              transform=ax.transAxes, va="top", ha="left")

            fig.canvas.mpl_connect("key_press_event", _on_key)
            plt.show()
            plt.close(fig)


    def plot_pc_timeseries_row(
        self,
        comparison: str,
        export_pdf: bool = True,
        use_cached_views: bool = False,
        pdf_name: str | None = None,
    ):
        """
        Build a single row of subplots (one per region) showing:
        - 3D projection line preview (PC1–PC3)
        - Small inset with Distance(t) and Angle(t)
        Defaults to the library's *default* 3D viewing angle. If `use_cached_views`
        is True, uses saved (elev, azim) from the JSON cache (when present).

        For 'int_nonint_face': draw time-encoded paths (white→blue and white→orange),
        each with an outline that matches the colormap hue (blue/orange).
        Only the first subplot shows a legend.
        """
        self._apply_illustrator_rc()

        labels = self._labels_for_comparison(comparison)
        df = self.psth_extractor.fetch_psth(comparison)
        regions = list(df["region"].unique())
        if not regions:
            logger.warning("No regions found to plot.")
            return

        cache = self._load_view_cache().get(comparison, {}) if use_cached_views else {}

        # Layout: 1 row, N columns
        fig = plt.figure(figsize=(4.6 * len(regions), 4.2))
        gs = fig.add_gridspec(1, len(regions))

        for j, region in enumerate(regions):
            region_df = df[df["region"] == region]
            proj_X1, proj_X2 = self._compute_combined_proj(region_df, *labels)
            timeline = self._timeline_from_proj(proj_X1)

            ax3d = fig.add_subplot(gs[0, j], projection="3d")
            elev_az = cache.get(str(region), None) if use_cached_views else None

            if comparison == "int_non_int_face":
                # Interactive = Blues; Non-interactive = Oranges; outline auto from cmap
                self._plot_time_encoded_line3d_with_outline(
                    ax3d, proj_X1, cmap="Blues", outline_color=None, label=labels[0]
                )
                self._plot_time_encoded_line3d_with_outline(
                    ax3d, proj_X2, cmap="Oranges", outline_color=None, label=labels[1]
                )
            else:
                # Simple solid lines
                ax3d.plot(proj_X1[:, 0], proj_X1[:, 1], proj_X1[:, 2], label=labels[0], color="tab:blue")
                ax3d.plot(proj_X2[:, 0], proj_X2[:, 1], proj_X2[:, 2], label=labels[1], color="tab:orange")

            ax3d.set_xlabel("PC1"); ax3d.set_ylabel("PC2"); ax3d.set_zlabel("PC3")
            ax3d.set_title(str(region))

            # Apply cached view ONLY if requested and available
            if elev_az is not None:
                ax3d.view_init(elev=elev_az["elev"], azim=elev_az["azim"])

            # Inset: distance & angle
            dist, angle_deg = self._distance_and_angle(proj_X1, proj_X2)
            bbox = ax3d.get_position()
            inset = fig.add_axes([bbox.x1 - 0.14, bbox.y1 - 0.18, 0.12, 0.14])
            inset.plot(timeline, dist, lw=1.2, label="Dist")
            inset.plot(timeline, angle_deg, lw=1.2, label="Angle")
            inset.set_xticks([]); inset.set_yticks([]); inset.set_frame_on(True)

            if j == 0:
                ax3d.legend(loc="upper left")
                if elev_az is not None:
                    ax3d.text2D(
                        0.02, 0.96,
                        f"elev={elev_az['elev']:.1f}, azim={elev_az['azim']:.1f}",
                        transform=ax3d.transAxes, va="top", ha="left", fontsize=9
                    )

        supt = f"{comparison} — PC trajectories (row) + distance/angle insets"
        fig.suptitle(supt, y=0.98)
        fig.tight_layout()

        if export_pdf:
            out = self.pca_config.trajectories_dir / (
                pdf_name if pdf_name else f"{comparison}__row_timeseries.pdf"
            )
            fig.savefig(out)
            logger.info(f"Exported: {out}")
        plt.close(fig)



    def plot_violin_distance_angle(self, comparison: str, export_pdf: bool = True):
        """
        Build two violins (Distance, Angle) across regions and show pairwise Welch
        p-values in dedicated text panels *below* each violin. Uses a 2×2 GridSpec
        to prevent clipping on export.
        """
        self._apply_illustrator_rc()

        labels = self._labels_for_comparison(comparison)
        df = self.psth_extractor.fetch_psth(comparison)

        per_region: Dict[str, Dict[str, np.ndarray]] = {}
        for region, region_df in df.groupby("region"):
            proj_X1, proj_X2 = self._compute_combined_proj(region_df, *labels)
            dist, angle_deg = self._distance_and_angle(proj_X1, proj_X2)
            per_region[str(region)] = {
                "distance": np.asarray(dist, dtype=float),
                "angle": np.asarray(angle_deg, dtype=float),
            }

        regions = list(per_region.keys())
        if not regions:
            logger.warning("No regions available for violin plots.")
            return

        dist_data = [per_region[r]["distance"] for r in regions]
        angle_data = [per_region[r]["angle"] for r in regions]

        from scipy.stats import ttest_ind
        def _pairwise_pvals(data: List[np.ndarray]):
            P = {}
            for i, j in combinations(range(len(data)), 2):
                xi, xj = data[i], data[j]
                if xi.size >= 2 and xj.size >= 2:
                    P[(i, j)] = ttest_ind(xi, xj, equal_var=False, nan_policy="omit").pvalue
                else:
                    P[(i, j)] = np.nan
            return P

        p_dist = _pairwise_pvals(dist_data)
        p_angle = _pairwise_pvals(angle_data)

        # --- Layout: 2 rows x 2 cols (violins on top; text panels on bottom) ---
        fig = plt.figure(figsize=(12, 6.4), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], wspace=0.25, hspace=0.25)

        ax_v_dist = fig.add_subplot(gs[0, 0])
        ax_v_ang  = fig.add_subplot(gs[0, 1])
        ax_t_dist = fig.add_subplot(gs[1, 0])
        ax_t_ang  = fig.add_subplot(gs[1, 1])

        # Violin plots (top row)
        self._violin_with_points(ax_v_dist, dist_data, regions, ylab="Distance")
        self._violin_with_points(ax_v_ang,  angle_data, regions, ylab="Angle (deg)")

        ax_v_dist.set_title("Distance across regions", fontsize=11)
        ax_v_ang.set_title("Angle across regions", fontsize=11)

        # Text panels (bottom row)
        def _panel_text(ax, title: str, pvals: Dict[Tuple[int, int], float], regions: List[str]):
            ax.axis("off")
            lines = []
            for (i, j), p in sorted(pvals.items()):
                if np.isnan(p):
                    line = f"{regions[i]} vs {regions[j]}: p = N/A"
                else:
                    line = f"{regions[i]} vs {regions[j]}: p = {p:.3g}"
                lines.append(line)
            txt = title + "\n" + ("\n".join(lines) if lines else "(no pairwise stats)")
            ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=9)

        _panel_text(ax_t_dist, "Pairwise Welch p-values (Distance):", p_dist, regions)
        _panel_text(ax_t_ang,  "Pairwise Welch p-values (Angle):",    p_angle, regions)

        fig.suptitle(f"{comparison} — Distance & Angle distributions across regions", y=0.995)

        if export_pdf:
            out = self.pca_config.evr_bars_dir / f"{comparison}__distance_angle_violins.pdf"
            # constrained_layout handles spacing; bbox_inches adds a tiny safety pad
            fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
            logger.info(f"Exported: {out}")
        plt.close(fig)



    # -------------------- Helpers / Internals --------------------

    def _apply_illustrator_rc(self):
        for k, v in self._ILLUSTRATOR_RCPARAMS.items():
            mpl.rcParams[k] = v

    def _ensure_dirs(self):
        self.pca_config.trajectories_dir.mkdir(parents=True, exist_ok=True)
        self.pca_config.evr_bars_dir.mkdir(parents=True, exist_ok=True)
        self.view_cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _labels_for_comparison(self, comparison: str) -> Tuple[str, str]:
        if comparison == "face_obj":
            return ("face", "object")
        elif comparison == "int_non_int_face":
            return ("face_interactive", "face_non_interactive")
        else:
            raise ValueError(f"Unknown comparison: {comparison}")

    def _compute_combined_proj(self, region_df: pd.DataFrame, label1: str, label2: str):
        """Build X1, X2, fit PCA on concatenated trials, return projections of X1, X2."""
        X1 = self._build_matrix(region_df, label1)
        X2 = self._build_matrix(region_df, label2)
        Xc = np.concatenate([X1, X2], axis=0)
        pca = PCA(n_components=self.pca_config.n_components).fit(Xc)
        return pca.transform(X1), pca.transform(X2)

    def _timeline_from_proj(self, proj: np.ndarray) -> np.ndarray:
        cfg = self.psth_extractor.config
        start, end = cfg.psth_window
        bs = cfg.psth_bin_size
        n = proj.shape[0]
        return np.linspace(start + bs / 2, end - bs / 2, n)

    def _distance_and_angle(self, proj_X1: np.ndarray, proj_X2: np.ndarray):
        # Optional mean-centering for angle as in parent class
        if getattr(self.pca_config, "mean_center_for_angle", False):
            X1c = proj_X1 - proj_X1.mean(axis=0, keepdims=True)
            X2c = proj_X2 - proj_X2.mean(axis=0, keepdims=True)
        else:
            X1c, X2c = proj_X1, proj_X2

        dist = np.linalg.norm(proj_X1 - proj_X2, axis=1)

        dot = np.sum(X1c * X2c, axis=1)
        n1 = np.linalg.norm(X1c, axis=1)
        n2 = np.linalg.norm(X2c, axis=1)
        angle = np.degrees(np.arccos(np.clip(dot / (n1 * n2 + 1e-10), -1.0, 1.0)))
        return dist, angle

    # ---- Time-encoded vector line with outline (3D) ----
    def _plot_time_encoded_line3d_with_outline(
        self,
        ax3d,
        pts: np.ndarray,
        cmap: str = "Blues",
        outline_color: Optional[str] = None,
        label: Optional[str] = None,
        lw_main: float = 2.2,
        lw_outline: float = 3.6,
    ):
        """
        Draw a polyline as many small segments:
        - An underlaid outline (using the dominant hue of the cmap)
        - A main segment collection colored by time (white -> target cmap)
        Stays vector on PDF/SVG (no rasterization).
        """
        import matplotlib as mpl

        # Downsample if too many points
        n = pts.shape[0]
        max_segs = 300
        if n - 1 > max_segs:
            keep = np.linspace(0, n - 1, max_segs + 1).astype(int)
            pts = pts[keep]

        t = np.linspace(0.0, 1.0, pts.shape[0])
        cmap_obj = plt.get_cmap(cmap)

        # Decide outline color
        if outline_color is None:
            # Take a saturated color from the cmap (near the end)
            outline_rgba = cmap_obj(0.95)
        else:
            outline_rgba = mpl.colors.to_rgba(outline_color)

        # Outline — thicker, solid
        for i in range(pts.shape[0] - 1):
            x = pts[i:i+2, 0]; y = pts[i:i+2, 1]; z = pts[i:i+2, 2]
            ax3d.plot(x, y, z, color=outline_rgba, lw=lw_outline, zorder=1)

        # Colored main line: white -> cmap
        for i in range(pts.shape[0] - 1):
            x = pts[i:i+2, 0]; y = pts[i:i+2, 1]; z = pts[i:i+2, 2]
            c_end = np.array(cmap_obj(t[i]))
            c = (1.0 - t[i]) * np.array([1, 1, 1, 1]) + t[i] * c_end
            ax3d.plot(x, y, z, color=c, lw=lw_main, zorder=2)

        if label:
            # Legend handle = solid outline color
            ax3d.plot([], [], [], color=outline_rgba, lw=lw_main, label=label)


    # ---- Violin helper (points + quartiles) ----
    def _violin_with_points(self, ax, data: List[np.ndarray], cats: List[str], ylab: str):
        parts = ax.violinplot(data, showmedians=False, showextrema=False)
        # Color each violin differently and add quartiles
        for i, b in enumerate(parts['bodies']):
            b.set_alpha(0.7)
            # simple distinct hues
            b.set_facecolor(plt.cm.tab10(i % 10))
            b.set_edgecolor("black")
            b.set_linewidth(0.8)

            vals = np.asarray(data[i], dtype=float)
            if vals.size:
                q1, q2, q3 = np.nanpercentile(vals, [25, 50, 75])
                ax.plot([i+1, i+1], [q1, q3], lw=2.0, color="black")  # IQR bar
                ax.scatter([i+1], [q2], s=16, zorder=3, edgecolor="black", facecolor="white")  # median

        # Overlay raw points (jittered)
        for i, vals in enumerate(data, start=1):
            if len(vals) == 0:
                continue
            x = np.random.normal(i, 0.03, size=len(vals))
            ax.scatter(x, vals, s=6, alpha=0.6, edgecolor="none")

        ax.set_xticks(range(1, len(cats)+1))
        ax.set_xticklabels([str(c) for c in cats], rotation=0)
        ax.set_ylabel(ylab)

    def _pval_summary(self, title: str, pvals: Dict[Tuple[int, int], float], regions: List[str]) -> str:
        # Short inline summary: e.g., "Distance — p(LIP vs OFC)=0.012, ..."
        pairs = []
        for (i, j), p in pvals.items():
            if np.isnan(p):
                continue
            pairs.append(f"{regions[i]} vs {regions[j]}: p={p:.3g}")
        if not pairs:
            return f"{title} — no pairwise stats"
        return f"{title} — " + ", ".join(pairs)

    # ---- View cache I/O ----
    def _load_view_cache(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        if self.view_cache_path.exists():
            try:
                return json.loads(self.view_cache_path.read_text())
            except Exception as e:
                logger.warning(f"Failed to read view cache: {e}")
        return {}

    def _save_view_cache(self, cache: Dict[str, Dict[str, Dict[str, float]]]):
        try:
            self.view_cache_path.write_text(json.dumps(cache, indent=2))
        except Exception as e:
            logger.error(f"Failed to write view cache: {e}")
