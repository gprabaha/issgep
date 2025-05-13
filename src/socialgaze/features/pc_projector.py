# src/socialgaze/features/pc_projector.py


import pdb
import logging
import os
import json
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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
            trialwise=fit_spec.trialwise,
            categories=fit_spec.categories,
            split_by_interactive=fit_spec.split_by_interactive
        )

        self.pc_fit_models[fit_spec.name] = {}
        self.unit_and_category_orders[fit_spec.name] = {}

        for region in df["region"].unique():
            region_df = df[df["region"] == region]
            pop_mat, unit_order, category_order, min_fixations, _ = self._build_population_matrix(region_df, fit_spec)
            
            pca = PCA(n_components=min(20, pop_mat.shape[0]))
            pca.fit(pop_mat)

            self.pc_fit_models[fit_spec.name][region] = pca
            self.unit_and_category_orders[fit_spec.name][region] = {
                "unit_order": unit_order,
                "category_order": category_order,
                "trials_per_category": min_fixations,
            }

            save_pickle(pca, get_pc_fit_model_path(self.config.pc_projection_base_dir, fit_spec.name, region))
            save_pickle(self.unit_and_category_orders[fit_spec.name][region],
                        get_pc_fit_orders_path(self.config.pc_projection_base_dir, fit_spec.name, region))

    def project(self, fit_spec_name: str, transform_spec: PCATransformSpec):
        logger.info(f"Projecting using fit: {fit_spec_name} and transform: {transform_spec.name}")
        
        df = self._get_filtered_psth_df(
            trialwise=transform_spec.trialwise,
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
            pop_mat, unit_order, category_order, min_fixations, sample_metadata = self._build_population_matrix(
                region_df, transform_spec
            )

            # Project and reshape
            projected = pca.transform(pop_mat)
            proj_df = self._reshape_projection_as_timeseries_dataframe(
                projected=projected,
                sample_metadata=sample_metadata,
                region=region,
                n_components=pca.n_components_,
                transform_spec=transform_spec,
                min_fixations=min_fixations
            )

            # Store and save
            self.pc_projection_dfs[key][region] = proj_df
            self.pc_projection_meta[key]["trials_per_category"] = min_fixations
            self.pc_projection_meta[key]["category_order"] = category_order

            save_df_to_pkl(
                proj_df,
                get_pc_projection_path(self.config.pc_projection_base_dir, fit_spec_name, transform_spec.name, region)
            )
            save_pickle(
                self.pc_projection_meta[key],
                get_pc_projection_meta_path(self.config.pc_projection_base_dir, fit_spec_name, transform_spec.name)
            )


    def _get_filtered_psth_df(self, trialwise, categories, split_by_interactive, agent=None):
        if trialwise:
            df = self.psth_extractor.get_psth("trial_wise")
        elif split_by_interactive:
            df = self.psth_extractor.get_psth("by_interactivity")
        else:
            df = self.psth_extractor.get_psth("by_category")

        if categories is not None:
            df = df[df["category"].isin(categories)]

        df = df[df["agent"] == (agent or "m1")]
        return df

    def _build_population_matrix(self, region_df, specs):
        trialwise = specs.trialwise
        split_by_interactive = specs.split_by_interactive
        base_categories = specs.categories or sorted(region_df["category"].unique())
        unit_uuids = sorted(region_df["unit_uuid"].unique())
        sample_metadata = []

        if not trialwise:
            if not split_by_interactive:
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
                return pop_mat, unit_uuids, category_order, None, sample_metadata

            else:
                interactive_conditions = ["interactive", "non-interactive"]
                category_order = [f"{cat}_{inter}" for cat in base_categories for inter in interactive_conditions]
                pop_list = []
                for unit_uuid in unit_uuids:
                    unit_frs = []
                    for cat in base_categories:
                        for inter in interactive_conditions:
                            row = region_df.query("unit_uuid == @unit_uuid and category == @cat and is_interactive == @inter")
                            if row.shape[0] != 1:
                                raise ValueError(f"Expected 1 row for {unit_uuid}, {cat}, {inter}, got {row.shape[0]}")
                            fr = np.array(row.iloc[0]["avg_firing_rate"])
                            for t in range(len(fr)):
                                sample_metadata.append({
                                    "agent": row.iloc[0]["agent"],
                                    "unit_uuid": unit_uuid,
                                    "category": cat,
                                    "interactivity": inter,
                                    "timepoint_index": t
                                })
                            unit_frs.append(fr)
                    pop_list.append(np.concatenate(unit_frs))
                pop_mat = np.stack(pop_list, axis=0).T
                return pop_mat, unit_uuids, category_order, None, sample_metadata

        else:
            interactive_conditions = ["interactive", "non-interactive"] if split_by_interactive else [None]
            all_keys = [(cat, inter) for cat in base_categories for inter in interactive_conditions]
            min_fixations = float("inf")

            for cat, inter in all_keys:
                if inter is None:
                    subset = region_df.query("category == @cat")
                else:
                    subset = region_df.query("category == @cat and is_interactive == @inter")
                
                grouped = subset.groupby("unit_uuid").size()
                if not grouped.empty:
                    min_fixations = min(min_fixations, grouped.min())

            if min_fixations == 0:
                raise ValueError("One or more (category, interaction, session) groups have zero fixations")

            category_order = [
                f"{cat}_{inter}" if inter is not None else cat
                for cat, inter in all_keys
            ]
            pop_list = []

            for unit_uuid in unit_uuids:
                unit_frs = []
                for cat, inter in all_keys:
                    if inter is None:
                        subset = region_df.query("category == @cat")
                    else:
                        subset = region_df.query("category == @cat and is_interactive == @inter")
                    if len(subset) < min_fixations:
                        raise ValueError(f"Not enough trials for {unit_uuid} in category {cat} (interactivity: {inter})")
                        
                    sampled = subset.sample(n=min_fixations, random_state=42)
                    for trial_index, (_, row) in enumerate(sampled.iterrows()):
                        fr = np.array(row["firing_rate"])
                        for t in range(len(fr)):
                            sample_metadata.append({
                                "session_name": row["session_name"],
                                "run_number": row["run_number"],
                                "agent": row["agent"],
                                "unit_uuid": unit_uuid,
                                "category": cat,
                                "interactivity": inter,
                                "trial_index": trial_index,
                                "timepoint_index": t
                            })
                        unit_frs.append(fr)
                pop_list.append(np.concatenate(unit_frs))
            pop_mat = np.stack(pop_list, axis=0).T
            return pop_mat, unit_uuids, category_order, min_fixations, sample_metadata


    def _reshape_projection_as_timeseries_dataframe(
        self,
        projected: np.ndarray,
        sample_metadata: List[Dict],
        region: str,
        n_components: int,
        transform_spec: PCATransformSpec,
        min_fixations: int
    ) -> pd.DataFrame:
        """
        Reshapes PCA projections into one row per PC dimension and condition, with timeseries as a vector.
        Assumes projected.shape[0] == n_timepoints × n_categories × n_interactivity × n_trials

        Args:
            projected: np.ndarray of shape (n_samples, n_components)
            sample_metadata: List of dicts describing each sample (should be repeated over units)
            region: brain region name
            n_components: number of PCA components
            transform_spec: used to check trialwise and interactivity
            min_fixations: number of trials per category (if trialwise), otherwise 1

        Returns:
            pd.DataFrame with one row per PC dimension per condition
        """
        # === Step 1: Infer unique timepoints
        unique_timepoints = sorted(set(meta["timepoint_index"] for meta in sample_metadata))
        n_timepoints = len(unique_timepoints)

        # === Step 2: Determine how many projected blocks we expect
        categories = transform_spec.categories or sorted(set(meta["category"] for meta in sample_metadata))
        n_categories = len(categories)
        n_interactive = 2 if transform_spec.split_by_interactive else 1
        n_trials = min_fixations if transform_spec.trialwise else 1

        expected_rows = n_timepoints * n_categories * n_interactive * n_trials
        assert projected.shape[0] == expected_rows, (
            f"Expected {expected_rows} rows in projected, got {projected.shape[0]} "
            f"(n_timepoints={n_timepoints}, n_categories={n_categories}, "
            f"n_interactive={n_interactive}, n_trials={n_trials})"
        )

        # === Step 3: Group by condition — one entry per condition, ignore units and timepoints
        grouped_keys = []
        seen_keys = set()
        for meta in sample_metadata:
            key = (
                meta["category"],
                meta.get("interactivity", None),
                meta.get("trial_index", None),
                meta.get("session_name", None),
                meta.get("run_number", None),
                meta.get("agent", None),
            )
            if key not in seen_keys:
                seen_keys.add(key)
                grouped_keys.append(key)

        # === Step 4: Sort keys (optional but ensures deterministic output)
        grouped_keys = sorted(grouped_keys)

        # === Step 5: Iterate and extract projections
        rows = []
        for i, (cat, inter, trial, session, run, agent) in enumerate(grouped_keys):
            start = i * n_timepoints
            end = start + n_timepoints
            matrix = projected[start:end]  # shape: (n_timepoints, n_components)

            for dim in range(n_components):
                rows.append({
                    "region": region,
                    "pc_dimension": dim,
                    "category": cat,
                    "is_interactive": inter,
                    "trial_index": trial,
                    "session": session,
                    "run": run,
                    "agent": agent,
                    "pc_timeseries": matrix[:, dim].tolist()
                })

        return pd.DataFrame(rows)





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

