# src/socialgaze/config/pca_config.py

from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.path_utils import get_pc_model_basedir


class PCAConfig(BaseConfig):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)

        # === PCA control flags ===
        self.use_parallel = True
        self.fit_categories = True
        self.fit_by_trial = False
        self.project_categories = True
        self.project_by_trial = False

        # === Category selection ===
        self.categories_to_include: list = None  # If None, use all

        # === Paths to save fits and projections ===
        self.pc_projection_base_dir = get_pc_model_basedir(self)


