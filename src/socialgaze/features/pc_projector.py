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

        self.pc_fit_models: Dict[str, Dict[str, PCA]] = {}
        self.pc_projection_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.pc_projection_meta: Dict[str, Dict] = {}

    def fit(self, fit_spec: PCAFitSpec):
        logger.info(f"Fitting PCA using: {fit_spec.name}")

        df = self._get_filtered_psth_df(
            trialwise=fit_spec.trialwise,
            categories=fit_spec.categories,
            split_by_interactive=fit_spec.split_by_interactive
        )

        self.pc_fit_models[fit_spec.name] = {}

        for region in df["region"].unique():
            region_df = df[df["region"] == region]
            pop_mat, sample_labels, unit_order = self._build_population_matrix_across_units(region_df, fit_spec)

            pca = PCA(n_components=min(20, pop_mat.shape[1]))
            pca.fit(pop_mat)

            self.pc_fit_models[fit_spec.name][region] = pca

            save_pickle(pca, get_pc_fit_model_path(self.config.pc_projection_base_dir, fit_spec.name, region))
            save_pickle({"unit_order": unit_order}, get_pc_fit_orders_path(self.config.pc_projection_base_dir, fit_spec.name, region))

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

            pca, order_meta = self.get_fit(fit_spec_name, region)
            pop_mat, sample_labels, _ = self._build_population_matrix_across_units(region_df, transform_spec, unit_order=order_meta["unit_order"])
            projected = pca.transform(pop_mat)

            proj_df = pd.DataFrame(projected, columns=[f"pc{i+1}" for i in range(projected.shape[1])])
            proj_df = pd.concat([sample_labels.reset_index(drop=True), proj_df], axis=1)
            self.pc_projection_dfs[key][region] = proj_df

            save_df_to_pkl(
                proj_df,
                get_pc_projection_path(self.config.pc_projection_base_dir, fit_spec_name, transform_spec.name, region)
            )
            with open(get_pc_projection_meta_path(self.config.pc_projection_base_dir, fit_spec_name, transform_spec.name), "w") as f:
                json.dump(self.pc_projection_meta[key], f, indent=2)

    def get_fit(self, fit_name: str, region: str) -> Tuple[PCA, Dict]:
        pca = self._load_or_get_fit_model(fit_name, region)
        orders = self._load_or_get_fit_orders(fit_name, region)
        return pca, orders

    def get_projection(self, fit_name: str, transform_name: str, region: str) -> Tuple[pd.DataFrame, Dict]:
        key = f"{fit_name}__{transform_name}"
        if key not in self.pc_projection_dfs:
            self.pc_projection_dfs[key] = {}
        if region not in self.pc_projection_dfs[key]:
            self.pc_projection_dfs[key][region] = self.load_projection(fit_name, transform_name, region)
        if key not in self.pc_projection_meta:
            self.pc_projection_meta[key] = self.load_projection_meta(fit_name, transform_name)
        return self.pc_projection_dfs[key][region], self.pc_projection_meta[key]

    def _get_filtered_psth_df(self, trialwise, categories, split_by_interactive, agent: Optional[str] = None):
        if trialwise:
            df = self.psth_extractor.get_psth("trial_wise")
        elif split_by_interactive:
            df = self.psth_extractor.get_psth("by_interactivity")
        else:
            df = self.psth_extractor.get_psth("by_category")

        if categories is not None:
            df = df[df["category"].isin(categories)]

        if "agent" in df.columns:
            if agent is not None:
                df = df[df["agent"] == agent]
            else:
                df = df[df["agent"] == "m1"]

        return df

    def _build_population_matrix_across_units(
        self,
        region_df: pd.DataFrame,
        specs: Union[PCAFitSpec, PCATransformSpec],
        unit_order: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
        trialwise = specs.trialwise
        split_by_interactive = specs.split_by_interactive
        base_categories = specs.categories or sorted(region_df["category"].unique())

        if unit_order is None:
            unit_order = sorted(region_df["unit_uuid"].unique())

        rows = []
        meta = []

        if not trialwise:
            for cat in base_categories:
                if split_by_interactive:
                    for inter in ["interactive", "non-interactive"]:
                        trial_df = region_df.query("category == @cat and is_interactive == @inter")
                        vec = []
                        for unit in unit_order:
                            sub = trial_df[trial_df["unit_uuid"] == unit]
                            if sub.empty:
                                raise ValueError(f"Missing data for {unit}, {cat}_{inter}")
                            vec.append(np.mean(sub.iloc[0]["avg_firing_rate"]))
                        rows.append(vec)
                        meta.append({"category": cat, "is_interactive": inter})
                else:
                    trial_df = region_df.query("category == @cat")
                    vec = []
                    for unit in unit_order:
                        sub = trial_df[trial_df["unit_uuid"] == unit]
                        if sub.empty:
                            raise ValueError(f"Missing data for {unit}, {cat}")
                        vec.append(np.mean(sub.iloc[0]["avg_firing_rate"]))
                    rows.append(vec)
                    meta.append({"category": cat})
        else:
            for cat in base_categories:
                if split_by_interactive:
                    for inter in ["interactive", "non-interactive"]:
                        trial_df = region_df.query("category == @cat and is_interactive == @inter")
                        grouped = trial_df.groupby(["session_name", "run_number", "fixation_start_idx"])
                        for _, group in grouped:
                            vec = []
                            for unit in unit_order:
                                sub = group[group["unit_uuid"] == unit]
                                if sub.empty:
                                    vec.append(0.0)
                                else:
                                    vec.append(np.mean(sub.iloc[0]["firing_rate"]))
                            rows.append(vec)
                            meta.append({"category": cat, "is_interactive": inter})
                else:
                    trial_df = region_df.query("category == @cat")
                    grouped = trial_df.groupby(["session_name", "run_number", "fixation_start_idx"])
                    for _, group in grouped:
                        vec = []
                        for unit in unit_order:
                            sub = group[group["unit_uuid"] == unit]
                            if sub.empty:
                                vec.append(0.0)
                            else:
                                vec.append(np.mean(sub.iloc[0]["firing_rate"]))
                        rows.append(vec)
                        meta.append({"category": cat})

        pop_mat = np.array(rows)
        sample_labels = pd.DataFrame(meta)
        return pop_mat, sample_labels, unit_order

    def _load_or_get_fit_model(self, fit_name: str, region: str) -> PCA:
        if fit_name not in self.pc_fit_models:
            self.pc_fit_models[fit_name] = {}
        if region not in self.pc_fit_models[fit_name]:
            path = get_pc_fit_model_path(self.config.pc_projection_base_dir, fit_name, region)
            self.pc_fit_models[fit_name][region] = load_pickle(path)
        return self.pc_fit_models[fit_name][region]

    def _load_or_get_fit_orders(self, fit_name: str, region: str) -> Dict:
        path = get_pc_fit_orders_path(self.config.pc_projection_base_dir, fit_name, region)
        return load_pickle(path)

    def load_projection(self, fit_name: str, transform_name: str, region: str):
        path = get_pc_projection_path(self.config.pc_projection_base_dir, fit_name, transform_name, region)
        return load_df_from_pkl(path)

    def load_projection_meta(self, fit_name: str, transform_name: str):
        path = get_pc_projection_meta_path(self.config.pc_projection_base_dir, fit_name, transform_name)
        with open(path, "r") as f:
            return json.load(f)

