# src/socialgaze/features/pc_projector.py

import logging
import os
import json
from typing import Optional, List, Dict, Tuple
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

        self.pc_fit_models: Dict[str, Dict[str, PCA]] = {}  # fit_name -> region -> PCA
        self.unit_and_category_orders: Dict[str, Dict[str, Dict]] = {}  # fit_name -> region -> order dict
        self.pc_projection_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}  # fit__transform -> region -> projection
        self.pc_projection_meta: Dict[str, Dict] = {}  # fit__transform -> metadata dict

    def fit(self, fit_spec: PCAFitSpec):
        logger.info(f"Fitting PCA using: {fit_spec.name}")

        df = self._get_filtered_psth_df(
            trialwise=fit_spec.trialwise,
            categories=fit_spec.categories,
            split_by_interactive=fit_spec.split_by_interactive,
            for_fit=True
        )

        self.pc_fit_models[fit_spec.name] = {}
        self.unit_and_category_orders[fit_spec.name] = {}

        for region in df["region"].unique():
            region_df = df[df["region"] == region]
            pop_mat, unit_order, category_order = self._build_population_matrix(region_df)

            pca = PCA(n_components=min(20, pop_mat.shape[0]))
            pca.fit(pop_mat)

            self.pc_fit_models[fit_spec.name][region] = pca
            self.unit_and_category_orders[fit_spec.name][region] = {
                "unit_order": unit_order,
                "category_order": category_order,
            }

            save_pickle(pca, get_pc_fit_model_path(self.config.pc_projection_base_dir, fit_spec.name, region))
            save_pickle(
                self.unit_and_category_orders[fit_spec.name][region],
                get_pc_fit_orders_path(self.config.pc_projection_base_dir, fit_spec.name, region)
            )

    def project(self, fit_spec_name: str, transform_spec: PCATransformSpec):
        logger.info(f"Projecting using fit: {fit_spec_name} and transform: {transform_spec.name}")

        df = self._get_filtered_psth_df(
            trialwise=transform_spec.trialwise,
            categories=transform_spec.categories,
            split_by_interactive=transform_spec.split_by_interactive,
            for_fit=False
        )

        key = f"{fit_spec_name}__{transform_spec.name}"
        self.pc_projection_dfs[key] = {}
        self.pc_projection_meta[key] = {"fit": fit_spec_name, "transform": transform_spec.name}

        for region in df["region"].unique():
            region_df = df[df["region"] == region]

            pca, orders = self.get_fit(fit_spec_name, region)

            pop_mat, unit_order, category_order = self._build_population_matrix(region_df, category_order=orders["category_order"])
            projected = pca.transform(pop_mat)

            proj_df = self._build_projection_dataframe(projected, unit_order, category_order)
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

    def _get_filtered_psth_df(self, trialwise, categories, split_by_interactive, for_fit):
        if trialwise:
            df = self.psth_extractor.get_psth("trial_wise")
        elif split_by_interactive:
            df = self.psth_extractor.get_psth("by_interactivity")
        else:
            df = self.psth_extractor.get_psth("by_category")

        if categories is not None:
            df = df[df["category"].isin(categories)]

        return df

    def _build_population_matrix(self, region_df: pd.DataFrame, category_order: Optional[List[str]] = None):
        unit_uuids = region_df["unit_uuid"].unique()
        categories = category_order or region_df["category"].unique().tolist()

        pop_list = []
        for uuid in unit_uuids:
            unit_frs = []
            for cat in categories:
                sub_df = region_df.query("unit_uuid == @uuid and category == @cat")
                if sub_df.empty:
                    raise ValueError(f"Missing category {cat} for unit {uuid}")
                fr = sub_df.iloc[0]["avg_firing_rate"] if "avg_firing_rate" in sub_df else sub_df.iloc[0]["firing_rate"]
                unit_frs.append(np.array(fr))
            pop_list.append(np.concatenate(unit_frs))

        pop_mat = np.stack(pop_list, axis=0)
        return pop_mat, list(unit_uuids), categories

    def _build_projection_dataframe(self, projected: np.ndarray, unit_order: List[str], category_order: List[str]) -> pd.DataFrame:
        rows = []
        n_units = len(unit_order)
        n_categories = len(category_order)
        pcs_per_cat = projected.shape[1] // n_categories

        for i, unit_uuid in enumerate(unit_order):
            for j, cat in enumerate(category_order):
                start, stop = j * pcs_per_cat, (j + 1) * pcs_per_cat
                rows.append({
                    "unit_uuid": unit_uuid,
                    "category": cat,
                    "pc_projection": projected[i, start:stop].tolist(),
                })
        return pd.DataFrame(rows)

    def _load_or_get_fit_model(self, fit_name: str, region: str) -> PCA:
        if fit_name not in self.pc_fit_models:
            self.pc_fit_models[fit_name] = {}
        if region not in self.pc_fit_models[fit_name]:
            path = get_pc_fit_model_path(self.config.pc_projection_base_dir, fit_name, region)
            self.pc_fit_models[fit_name][region] = load_pickle(path)
        return self.pc_fit_models[fit_name][region]

    def _load_or_get_fit_orders(self, fit_name: str, region: str) -> Dict:
        if fit_name not in self.unit_and_category_orders:
            self.unit_and_category_orders[fit_name] = {}
        if region not in self.unit_and_category_orders[fit_name]:
            path = get_pc_fit_orders_path(self.config.pc_projection_base_dir, fit_name, region)
            self.unit_and_category_orders[fit_name][region] = load_pickle(path)
        return self.unit_and_category_orders[fit_name][region]

    def load_projection(self, fit_name: str, transform_name: str, region: str):
        path = get_pc_projection_path(self.config.pc_projection_base_dir, fit_name, transform_name, region)
        return load_df_from_pkl(path)

    def load_projection_meta(self, fit_name: str, transform_name: str):
        path = get_pc_projection_meta_path(self.config.pc_projection_base_dir, fit_name, transform_name)
        with open(path, "r") as f:
            return json.load(f)
