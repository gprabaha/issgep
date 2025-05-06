# src/socialgaze/config/pca_config.py

from socialgaze.config.base_config import BaseConfig
import os

class PCAConfig(BaseConfig):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)

        # === PCA control flags ===
        self.fit_categories = True
        self.fit_by_trial = False
        self.project_categories = True
        self.project_by_trial = False

        # === Category selection ===
        self.categories_to_include: list = None  # If None, use all

        # === Paths to save fits and projections ===
        self.pc_projection_base_dir = os.path.join(self.processed_data_dir, "pc_projection")

        # self.pc_projection_by_category_path = os.path.join(self.processed_data_dir, "pc_projection_by_category.pkl")
        # self.pc_projection_by_trial_path = os.path.join(self.processed_data_dir, "pc_projection_by_trial.pkl")
        # self.pc_fit_by_category_path = os.path.join(self.processed_data_dir, "pc_fit_by_category.pkl")
        # self.pc_fit_by_trial_path = os.path.join(self.processed_data_dir, "pc_fit_by_trial.pkl")
        # self.pc_orders_path = os.path.join(self.processed_data_dir, "pc_orders.pkl")
