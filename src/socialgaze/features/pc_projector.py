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


