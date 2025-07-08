# src/socialgaze/features/pc_projector.py

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
            logger.info(f"PCA for region {region}: face vs object")
            X_face = self._build_matrix(region_df, "face")
            X_object = self._build_matrix(region_df, "object")
            self._run_pca_analysis(X_face, X_object, region, "face", "object", comparison)


    def run_int_nonint_face_analysis(self):
        df = self.psth_extractor.fetch_psth("int_non_int_face")
        comparison = "int_nonint_face"
        for region, region_df in df.groupby("region"):
            logger.info(f"PCA for region {region}: face_interactive vs face_non_interactive")
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
        pca_comb = PCA(n_components=n_components).fit(X_comb)
        evr_comb_X1 = self._explained_var_on_data(pca_comb, X1)
        evr_comb_X2 = self._explained_var_on_data(pca_comb, X2)

        logger.info(f"[{region}] Fit {label1} → EVR on {label1}: {evr_X1_on_X1}")
        logger.info(f"[{region}] Fit {label1} → EVR on {label2}: {evr_X1_on_X2}")
        logger.info(f"[{region}] Fit {label2} → EVR on {label2}: {evr_X2_on_X2}")
        logger.info(f"[{region}] Fit {label2} → EVR on {label1}: {evr_X2_on_X1}")
        logger.info(f"[{region}] Fit combined → EVR on {label1}: {evr_comb_X1}")
        logger.info(f"[{region}] Fit combined → EVR on {label2}: {evr_comb_X2}")

        self._plot_evr_bar(
            region, evr_X1_on_X1, evr_X1_on_X2, evr_X2_on_X2, evr_X2_on_X1,
            evr_comb_X1, evr_comb_X2, label1, label2, comparison
        )

        proj_X1 = pca_comb.transform(X1)
        proj_X2 = pca_comb.transform(X2)
        self._plot_trajectories_with_metrics(proj_X1, proj_X2, label1, label2, region, comparison)


    def _explained_var_on_data(self, pca: PCA, X: np.ndarray) -> np.ndarray:
        total_var = np.sum(np.var(X, axis=0))
        X_proj = pca.transform(X)
        var_proj = np.var(X_proj, axis=0)
        return var_proj / total_var


    def _plot_evr_bar(
        self, region, evr_X1_X1, evr_X1_X2, evr_X2_X2, evr_X2_X1, evr_comb_X1, evr_comb_X2,
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
            f"{label2} EVR": evr_X2_X2,
            f"{label1} EVR": evr_X2_X1
        }, index=pcs)
        df2.plot(kind="bar", ax=axs[1])
        axs[1].set_title(f"Fit {label2}")
        axs[1].set_xlabel("PC")

        # Subplot 3: Fit both
        df3 = pd.DataFrame({
            f"{label1} EVR": evr_comb_X1,
            f"{label2} EVR": evr_comb_X2
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

        dist = np.linalg.norm(proj_X1 - proj_X2, axis=1)
        dot = np.sum(proj_X1 * proj_X2, axis=1)
        norm1 = np.linalg.norm(proj_X1, axis=1)
        norm2 = np.linalg.norm(proj_X2, axis=1)
        angle_deg = np.degrees(np.arccos(np.clip(dot / (norm1 * norm2 + 1e-10), -1.0, 1.0)))

        fig = plt.figure(figsize=(14, 10))

        # === 3D PC trajectory ===
        ax1 = fig.add_subplot(2, 1, 1, projection='3d')
        ax1.plot(proj_X1[:, 0], proj_X1[:, 1], proj_X1[:, 2], label=label1, color='blue')
        ax1.plot(proj_X2[:, 0], proj_X2[:, 1], proj_X2[:, 2], label=label2, color='red')
        ax1.set_title(f"{region} — {comparison} — PC1-PC2-PC3 Trajectory")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.set_zlabel("PC3")
        ax1.legend()

        # === Distance & angle ===
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(timeline, dist, label="Euclidean Distance", color='black')
        ax2.plot(timeline, angle_deg, label="Angle (deg)", color='green')
        ax2.set_title(f"{region} — {comparison} — Distance & Angle")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Value")
        ax2.legend()

        plt.tight_layout()

        out_path = self.pca_config.trajectories_dir / f"{comparison}__{region}.png"
        plt.savefig(out_path)
        logger.info(f"Saved 3D trajectory + metrics plot: {out_path}")
        plt.close()






















'''
import pdb
import logging
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial import procrustes

from os import listdir
from os.path import join, isdir

from socialgaze.utils.saving_utils import save_df_to_pkl, save_pickle
from socialgaze.utils.loading_utils import load_df_from_pkl, load_pickle
from socialgaze.utils.path_utils import (
    get_pc_fit_model_path,
    get_pc_fit_orders_path,
    get_pc_projection_path,
    get_pc_projection_meta_path,
)
from socialgaze.specs.pca_specs import PCAFitSpec, PCATransformSpec

logger = logging.getLogger(__name__)


class PCProjector:
    def __init__(self, config, psth_extractor):
        self.config = config
        self.psth_extractor = psth_extractor

        self.pc_fit_models = {}
        self.unit_and_category_orders = {}
        self.pc_projection_dfs = {}
        self.pc_projection_meta = {}

    def fit(self, fit_spec: PCAFitSpec):
        logger.info(f"Fitting PCA using: {fit_spec.name}")
        df = self._get_filtered_psth_df(
            categories=fit_spec.categories,
            split_by_interactive=fit_spec.split_by_interactive
        )

        self.pc_fit_models[fit_spec.name] = {}
        self.unit_and_category_orders[fit_spec.name] = {}

        for region in df["region"].unique():
            region_df = df[df["region"] == region]
            pop_mat, unit_order, category_order, sample_metadata = self._build_population_matrix(region_df, fit_spec)
            
            pca = PCA(n_components=min(20, pop_mat.shape[0]))
            pca.fit(pop_mat)

            self.pc_fit_models[fit_spec.name][region] = pca
            self.unit_and_category_orders[fit_spec.name][region] = {
                "unit_order": unit_order,
                "category_order": category_order,
            }

            save_pickle(pca, get_pc_fit_model_path(self.config.pc_projection_base_dir, fit_spec.name, region))
            save_pickle(self.unit_and_category_orders[fit_spec.name][region],
                        get_pc_fit_orders_path(self.config.pc_projection_base_dir, fit_spec.name, region))

    def project(self, fit_spec_name: str, transform_spec: PCATransformSpec):
        logger.info(f"Projecting using fit: {fit_spec_name} and transform: {transform_spec.name}")
        
        df = self._get_filtered_psth_df(
            categories=transform_spec.categories,
            split_by_interactive=transform_spec.split_by_interactive
        )

        key = f"{fit_spec_name}__{transform_spec.name}"
        self.pc_projection_dfs[key] = {}
        self.pc_projection_meta[key] = {
            "fit": fit_spec_name,
            "transform": transform_spec.name
        }

        for region in df["region"].unique():
            region_df = df[df["region"] == region]

            # Get fitted PCA model and condition orders
            pca, orders = self.get_fit(fit_spec_name, region)

            # Build population matrix and metadata
            pop_mat, unit_order, category_order, sample_metadata = self._build_population_matrix(region_df, transform_spec)

            # Project and reshape
            projected = pca.transform(pop_mat)
            proj_df = self._reshape_projection_as_timeseries_dataframe(
                projected=projected,
                sample_metadata=sample_metadata,
                region=region,
                n_components=pca.n_components_,
                transform_spec=transform_spec,
            )

            per_pc_var_explained = compute_per_pc_explained_variance_per_category(
                pca, region_df, unit_order, category_order, self.config.normalize_variance_within_category
            )
            
            # Store and save
            self.pc_projection_dfs[key][region] = proj_df
            self.pc_projection_meta[key]["category_order"] = category_order
            self.pc_projection_meta[key][f"{region}_category_pc_var_explained"] = per_pc_var_explained

            save_df_to_pkl(
                proj_df,
                get_pc_projection_path(self.config.pc_projection_base_dir, fit_spec_name, transform_spec.name, region)
            )
            save_pickle(
                self.pc_projection_meta[key],
                get_pc_projection_meta_path(self.config.pc_projection_base_dir, fit_spec_name, transform_spec.name)
            )

    def get_fit(self, fit_name, region):
        pca = self._load_or_get_fit_model(fit_name, region)
        orders = self._load_or_get_fit_orders(fit_name, region)
        return pca, orders

    def get_projection(self, fit_name, transform_name, region):
        key = f"{fit_name}__{transform_name}"
        if key not in self.pc_projection_dfs:
            self.pc_projection_dfs[key] = {}
        if region not in self.pc_projection_dfs[key]:
            self.pc_projection_dfs[key][region] = self.load_projection(fit_name, transform_name, region)
        if key not in self.pc_projection_meta:
            self.pc_projection_meta[key] = self.load_projection_meta(fit_name, transform_name)
        return self.pc_projection_dfs[key][region], self.pc_projection_meta[key]


    def get_available_fit_transform_region_keys(self) -> Dict[str, List[str]]:
        """
        Scans the saved projection directories and returns a dictionary mapping
        (fit_name, transform_name) to a list of regions with available projections.

        Returns:
            Dict[str, List[str]]: key = f"{fit_name}__{transform_name}", value = list of regions
        """
        root = self.config.pc_projection_base_dir
        keys_to_regions = {}

        for key in listdir(root):
            full_path = join(root, key)
            if not isdir(full_path) or "__" not in key:
                continue
            try:
                fit_name, transform_name = key.split("__")
            except ValueError:
                continue  # Skip malformed folders

            regions = []
            for fname in listdir(full_path):
                if fname.startswith("projection_") and fname.endswith(".pkl"):
                    region = fname.replace("projection_", "").replace(".pkl", "")
                    regions.append(region)

            if regions:
                keys_to_regions[key] = regions

        return keys_to_regions


    def compare_category_trajectories(self, fit_name: str, transform_name: str, region: str):
        """
        Computes distance metrics between all category trajectories in PCA-projected space:
        - Euclidean distance
        - Mean vector angle
        - Trajectory length difference
        - Procrustes disparity

        Returns:
            List[Dict] with keys:
                'category_1', 'category_2', 
                'euclidean_distance', 'vector_angle_deg',
                'trajectory_length_diff', 'procrustes_disparity'
        """
        proj_df, _ = self.get_projection(fit_name, transform_name, region)
        results = []

        categories = sorted(proj_df["category"].unique())

        # Build: category -> matrix (shape: (n_components, n_timepoints))
        cat_to_matrix = {}
        for cat in categories:
            cat_df = proj_df[proj_df["category"] == cat]
            matrices = [np.array(row["pc_timeseries"]) for _, row in cat_df.sort_values("pc_dimension").iterrows()]
            mat = np.stack(matrices)  # shape: (n_components, n_timepoints)
            cat_to_matrix[cat] = mat

        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat1, cat2 = categories[i], categories[j]
                mat1 = cat_to_matrix[cat1]
                mat2 = cat_to_matrix[cat2]

                # Euclidean distance
                eucl_dist = np.linalg.norm(mat1 - mat2)

                # Mean angle (transpose to (T, N))
                angle_rad = mean_vector_angle(mat1.T, mat2.T)
                angle_deg = np.degrees(angle_rad)

                # Trajectory lengths (along time)
                len1 = np.sum(np.linalg.norm(np.diff(mat1.T, axis=0), axis=1))
                len2 = np.sum(np.linalg.norm(np.diff(mat2.T, axis=0), axis=1))
                traj_len_diff = len1 - len2

                # Procrustes disparity
                try:
                    _, _, disparity = procrustes(mat1.T, mat2.T)
                except Exception as e:
                    logger.warning(f"Procrustes failed for {cat1} vs {cat2}: {e}")
                    disparity = np.nan

                results.append({
                    "category_1": cat1,
                    "category_2": cat2,
                    "euclidean_distance": eucl_dist,
                    "vector_angle_deg": angle_deg,
                    "trajectory_length_diff": traj_len_diff,
                    "procrustes_disparity": disparity
                })

        return results


    def _get_filtered_psth_df(self, categories, split_by_interactive, agent=None):
        """
        Returns a filtered PSTH DataFrame based on specified categories, interactivity split, and agent.

        If split_by_interactive is True and 'face' is in categories, then we search for
        'face_interactive' and 'face_non_interactive' in the already-split DataFrame.
        """
        if split_by_interactive:
            df = self.psth_extractor.get_psth("by_interactivity")

            # Expand categories to match split naming
            if categories is not None:
                expanded_categories = []
                for cat in categories:
                    if cat == "face":
                        expanded_categories.extend(["face_interactive", "face_non_interactive"])
                    else:
                        expanded_categories.append(cat)
                categories = expanded_categories

        else:
            df = self.psth_extractor.get_psth("by_category")

        if categories is not None:
            df = df[df["category"].isin(categories)]

        df = df[df["agent"] == (agent or "m1")]
        return df


    def _build_population_matrix(self, region_df, specs):
        """
        Builds the population matrix used for PCA fitting or projection.
        Uses avg_firing_rate columns grouped by unit_uuid and category.
        """
        split_by_interactive = specs.split_by_interactive
        base_categories = specs.categories or sorted(region_df["category"].unique())
        assert all(cat in {"face", "object", "out_of_roi"} for cat in base_categories)

        unit_uuids = sorted(region_df["unit_uuid"].unique())
        sample_metadata = []

        # If interactivity is split, categories already include it (e.g., "face_interactive")
        if split_by_interactive:
            category_order = []
            for cat in base_categories:
                if cat == "face":
                    category_order.extend(["face_interactive", "face_non_interactive"])
                else:
                    category_order.append(cat)
        else:
            category_order = base_categories

        pop_list = []
        for unit_uuid in unit_uuids:
            unit_frs = []
            for cat in category_order:
                row = region_df.query("unit_uuid == @unit_uuid and category == @cat")

                if row.shape[0] != 1:
                    raise ValueError(f"Expected 1 row for {unit_uuid}, {cat}, got {row.shape[0]}")

                fr = np.array(row.iloc[0]["avg_firing_rate"])
                for t in range(len(fr)):
                    sample_metadata.append({
                        "agent": row.iloc[0]["agent"],
                        "unit_uuid": unit_uuid,
                        "category": cat,
                        "timepoint_index": t,
                    })
                unit_frs.append(fr)

            pop_list.append(np.concatenate(unit_frs))

        pop_mat = np.stack(pop_list, axis=0).T
        return pop_mat, unit_uuids, category_order, sample_metadata


    def _reshape_projection_as_timeseries_dataframe(
        self,
        projected: np.ndarray,
        sample_metadata: List[Dict],
        region: str,
        n_components: int,
        transform_spec: PCATransformSpec,
    ) -> pd.DataFrame:
        """
        Reshapes PCA projections into one row per PC dimension and condition, with timeseries as a vector.
        Assumes projected.shape[0] == n_timepoints x n_unit x n_category

        Args:
            projected: np.ndarray of shape (n_samples, n_components)
            sample_metadata: List of dicts describing each sample (one per timepoint x unit x category)
            region: brain region name
            n_components: number of PCA components
            transform_spec: category and interactivity spec used for filtering

        Returns:
            pd.DataFrame with one row per PC dimension per category
        """
        # Step 1: Extract unique timepoints
        unique_timepoints = sorted(set(meta["timepoint_index"] for meta in sample_metadata))
        n_timepoints = len(unique_timepoints)

        # Step 2: Extract unique (category, agent, session, run) combinations
        grouped_keys = []
        seen_keys = set()
        for meta in sample_metadata:
            key = (
                meta["category"],
                meta.get("agent"),
            )
            if key not in seen_keys:
                seen_keys.add(key)
                grouped_keys.append(key)

        # Ensure deterministic ordering
        grouped_keys = sorted(grouped_keys)

        # Step 3: Sanity check shape
        expected_rows = len(grouped_keys) * n_timepoints
        assert projected.shape[0] == expected_rows, (
            f"Expected {expected_rows} rows in projected, got {projected.shape[0]} "
            f"(n_timepoints={n_timepoints}, n_conditions={len(grouped_keys)})"
        )

        # Step 4: Reshape and build final dataframe
        rows = []
        for i, (cat, agent) in enumerate(grouped_keys):
            start = i * n_timepoints
            end = start + n_timepoints
            matrix = projected[start:end]  # shape: (n_timepoints, n_components)

            for dim in range(n_components):
                rows.append({
                    "region": region,
                    "pc_dimension": dim,
                    "category": cat,
                    "agent": agent,
                    "pc_timeseries": matrix[:, dim].tolist(),
                })

        return pd.DataFrame(rows)



    def _load_or_get_fit_model(self, fit_name, region):
        if fit_name not in self.pc_fit_models:
            self.pc_fit_models[fit_name] = {}
        if region not in self.pc_fit_models[fit_name]:
            path = get_pc_fit_model_path(self.config.pc_projection_base_dir, fit_name, region)
            self.pc_fit_models[fit_name][region] = load_pickle(path)
        return self.pc_fit_models[fit_name][region]


    def _load_or_get_fit_orders(self, fit_name, region):
        if fit_name not in self.unit_and_category_orders:
            self.unit_and_category_orders[fit_name] = {}
        if region not in self.unit_and_category_orders[fit_name]:
            path = get_pc_fit_orders_path(self.config.pc_projection_base_dir, fit_name, region)
            self.unit_and_category_orders[fit_name][region] = load_pickle(path)
        return self.unit_and_category_orders[fit_name][region]


    def load_projection(self, fit_name, transform_name, region):
        path = get_pc_projection_path(self.config.pc_projection_base_dir, fit_name, transform_name, region)
        return load_df_from_pkl(path)

    def load_projection_meta(self, fit_name, transform_name):
        path = get_pc_projection_meta_path(self.config.pc_projection_base_dir, fit_name, transform_name)
        return load_df_from_pkl(path)


def compute_per_pc_explained_variance_per_category(
    pca,
    region_df,
    unit_order,
    category_order,
    normalize_within_category: bool = True
):
    """
    For each category, computes variance explained by each PC individually.

    Args:
        pca: fitted PCA model
        region_df: PSTH dataframe for one brain region
        unit_order: list of unit UUIDs (same order as used in PCA fitting)
        category_order: list of categories (should match PCA fitting order)
        normalize_within_category: 
            - True: normalize variance explained within each category
            - False: normalize all residuals using global total variance (shared across categories)

    Returns:
        Dict[str, List[float]]: category -> list of variance explained per PC
    """
    category_pc_var_explained = {}
    n_components = pca.n_components_

    # === Step 1: Compute global variance matrix (optional) ===
    if not normalize_within_category:
        pop_list = []
        for unit_uuid in unit_order:
            unit_frs = []
            for cat in category_order:
                row = region_df.query("unit_uuid == @unit_uuid and category == @cat")
                if row.shape[0] != 1:
                    raise ValueError(f"Expected 1 row for {unit_uuid}, {cat}, got {row.shape[0]}")
                fr = np.array(row.iloc[0]["avg_firing_rate"])
                unit_frs.append(fr)
            pop_list.append(np.concatenate(unit_frs))  # shape: (T * C,)
        X_all = np.stack(pop_list, axis=0).T  # shape: (samples, units)
        global_total_var = np.sum((X_all - pca.mean_) ** 2) + np.finfo(float).eps

    # === Step 2: Per-category analysis ===
    for cat in category_order:
        unit_frs = []
        for unit_uuid in unit_order:
            row = region_df.query("unit_uuid == @unit_uuid and category == @cat")
            if row.shape[0] != 1:
                raise ValueError(f"Expected 1 row for {unit_uuid}, {cat}, got {row.shape[0]}")
            fr = np.array(row.iloc[0]["avg_firing_rate"])
            unit_frs.append(fr)
        X_cat = np.stack(unit_frs, axis=0).T  # shape: (T, N) ← same as PCA input shape

        # Project and optionally compute per-category total variance
        X_proj = pca.transform(X_cat)

        if normalize_within_category:
            total_var = np.sum((X_cat - pca.mean_) ** 2) + np.finfo(float).eps
        else:
            total_var = global_total_var

        # Reconstruct using each PC separately
        pc_var_explained = []
        for i in range(n_components):
            X_proj_i = np.zeros_like(X_proj)
            X_proj_i[:, i] = X_proj[:, i]
            X_recon_i = pca.inverse_transform(X_proj_i)
            residual = X_cat - X_recon_i
            residual_var = np.sum(residual ** 2)

            explained = 1 - residual_var / total_var
            if not (0.0 <= explained <= 1.0):
                raise ValueError(
                    f"Explained variance for PC {i} in category '{cat}' was {explained:.4f}, "
                    f"which is outside the valid range [0, 1]."
                )

            pc_var_explained.append(explained)

        category_pc_var_explained[cat] = pc_var_explained

    return category_pc_var_explained


def mean_vector_angle(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the mean angle (in radians) between corresponding vectors in h1 and h2.
    First mean-shifts both trajectories across time.

    Args:
        h1, h2 (np.ndarray): shape (T, N)

    Returns:
        float: mean angular difference (radians)
    """
    if h1.shape != h2.shape:
        raise ValueError("h1 and h2 must have the same shape")

    # Mean shift along time (axis 0)
    h1_centered = h1 - h1.mean(axis=0, keepdims=True)
    h2_centered = h2 - h2.mean(axis=0, keepdims=True)

    dot_products = np.sum(h1_centered * h2_centered, axis=1)
    norms_h1 = np.linalg.norm(h1_centered, axis=1)
    norms_h2 = np.linalg.norm(h2_centered, axis=1)

    denom = norms_h1 * norms_h2
    denom[denom == 0] = np.finfo(float).eps

    cos_sim = dot_products / denom
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    angles = np.arccos(cos_sim)
    return np.mean(angles)

'''