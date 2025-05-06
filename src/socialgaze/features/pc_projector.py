# src/socialgaze/features/pc_projector.py

import logging
import glob
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import pdb

logger = logging.getLogger(__name__)


class PCProjector:
    def __init__(self, config, psth_extractor):
        """
        Initializes the PCProjector object.

        Args:
            config: PCAConfig object containing PCA parameters.
            psth_extractor: PSTHExtractor object with PSTH dataframes.
        """
        self.config = config
        self.psth_extractor = psth_extractor

        self.pc_fit_models: Dict[str, PCA] = {}  # region -> fitted PCA
        self.pc_projection_dfs: Dict[str, pd.DataFrame] = {}  # region -> dataframe of projections
        self.unit_and_category_orders: Dict[str, Dict[str, List[str]]] = {}  # region -> {'unit_order': [...], 'category_order': [...]}

    def project_avg_firing_rate_by_category(self):
        """
        Projects avg firing rates by category and saves the result in a DataFrame structure.
        """
        logger.info("Fetching avg PSTH by category...")
        # df = self.psth_extractor.get_psth(which="by_category")
        df = self.psth_extractor.get_psth(which="trial_wise")
        pdb.set_trace()

        categories_to_use = self.config.categories_to_include
        if categories_to_use is None:
            categories_to_use = df["category"].unique().tolist()

        logger.info(f"Using categories: {categories_to_use}")

        df = df[df["category"].isin(categories_to_use)]

        if "region" not in df.columns:
            raise ValueError("Region column is missing from the avg_psth_per_category dataframe.")

        regions = df["region"].unique()

        for region in regions:
            logger.info(f"Processing region: {region}")
            region_df = df[df["region"] == region]

            # Validate: Each (uuid, category) must appear exactly once
            if region_df.groupby(["unit_uuid", "category"]).size().gt(1).any():
                raise ValueError(f"Duplicate entries found for region {region}. Each (unit_uuid, category) must be unique.")

            # Build pop matrix
            pop_mat, unit_order, category_order = self._build_population_matrix(region_df, categories_to_use)

            # Fit PCA
            pca = PCA(n_components=min(20, pop_mat.shape[0]))
            projected = pca.fit_transform(pop_mat)

            self.pc_fit_models[region] = pca
            self.unit_and_category_orders[region] = {
                "unit_order": unit_order,
                "category_order": category_order,
            }

            # Build DataFrame for projection
            projection_df = self._build_projection_dataframe(projected, unit_order, category_order)

            self.pc_projection_dfs[region] = projection_df

            logger.info(f"Finished PCA fit and projection for region {region}")

    def _build_population_matrix(self, region_df: pd.DataFrame, categories: List[str]) -> (np.ndarray, List[str], List[str]):
        """
        Build the full matrix for PCA.

        Returns:
            pop_mat: (n_units x timepoints * n_categories)
            unit_order: order of uuids
            category_order: categories
        """
        unit_uuids = region_df["unit_uuid"].unique()
        pop_list = []

        for uuid in unit_uuids:
            unit_frs = []
            for cat in categories:
                sub_df = region_df.query("unit_uuid == @uuid and category == @cat")
                if sub_df.empty:
                    raise ValueError(f"Missing category {cat} for unit {uuid}")
                unit_frs.append(np.array(sub_df.iloc[0]["avg_firing_rate"]))
            unit_concat = np.concatenate(unit_frs, axis=0)
            pop_list.append(unit_concat)

        pop_mat = np.stack(pop_list, axis=0)
        return pop_mat, list(unit_uuids), categories

    def _build_projection_dataframe(self, projected: np.ndarray, unit_order: List[str], category_order: List[str]) -> pd.DataFrame:
        """
        Converts projected PC matrix into a structured dataframe.

        Returns:
            pd.DataFrame
        """
        rows = []
        n_units = len(unit_order)
        n_categories = len(category_order)

        pcs_per_category = projected.shape[1] // n_categories

        for i, unit_uuid in enumerate(unit_order):
            for j, category in enumerate(category_order):
                start = j * pcs_per_category
                stop = (j + 1) * pcs_per_category
                pc_segment = projected[i, start:stop]
                rows.append({
                    "unit_uuid": unit_uuid,
                    "category": category,
                    "pc_projection": pc_segment.tolist()
                })

        df = pd.DataFrame(rows)
        return df

    def save_dataframes(self):
        """
        Saves projection dataframes and orders.
        """
        logger.info("Saving PC projections and unit/category orders...")
        for region, df in self.pc_projection_dfs.items():
            save_df_to_pkl(df, self.config.pc_projection_by_category_path.replace(".pkl", f"_{region}.pkl"))
        save_df_to_pkl(self.unit_and_category_orders, self.config.pc_orders_path)
        logger.info("Saved PC projections and orders.")

    def load_dataframes(self):
        """
        Loads projection dataframes and orders.
        """
        logger.info("Loading PC projections and unit/category orders...")
        projection_paths = glob.glob(self.config.pc_projection_by_category_path.replace(".pkl", "_*.pkl"))
        for path in projection_paths:
            region = path.split("_")[-1].replace(".pkl", "")
            self.pc_projection_dfs[region] = load_df_from_pkl(path)

        self.unit_and_category_orders = load_df_from_pkl(self.config.pc_orders_path)
        logger.info("Loaded PC projections and orders.")
