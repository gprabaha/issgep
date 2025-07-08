# src/socialgaze/config/pca_config.py

from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.path_utils import PCAPaths


class PCAConfig(BaseConfig):
    """
    Configuration for population PCA.
    """

    def __init__(self, config_path: str = None):
        super().__init__(config_path)

        self.n_components = 20

        # Paths manager
        self.pca_paths = PCAPaths(self)

        # Final output plot dirs
        self.trajectories_dir = self.pca_paths.get_trajectories_dir()
        self.evr_bars_dir = self.pca_paths.get_evr_bars_dir()




'''
import os

from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.path_utils import (
    get_pc_model_basedir,
    get_pc_trajectory_plots_base_dir,
    get_pc_trajectory_plot_dir_for_fit_transform_combination,
    get_pc_trajectory_comparison_plots_base_dir
)


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
        self.normalize_variance_within_category: bool = False

        # === Paths to save fits and projections ===
        self.pc_projection_base_dir = get_pc_model_basedir(self)
        self.pc_trajectory_plot_base_dir = get_pc_trajectory_plots_base_dir(self)
        self.pc_trajectory_comparison_plots_base_dir = get_pc_trajectory_comparison_plots_base_dir(self)
        


    def get_static_pc_plot_path(self, fit_name: str, transform_name: str, region: str, plot_file_format: str, include_date: bool = True) -> str:
        base = get_pc_trajectory_plot_dir_for_fit_transform_combination(
            base_dir=self.pc_trajectory_plot_base_dir,
            fit_name=fit_name,
            transform_name=transform_name,
            dated=include_date
        )
        fname = f"all_conditions_{region}.{plot_file_format}"
        return os.path.join(base, fname)


    def get_rotation_pc_plot_path(self, fit_name: str, transform_name: str, region: str, include_date: bool = True) -> str:
        base = get_pc_trajectory_plot_dir_for_fit_transform_combination(
            base_dir=self.pc_trajectory_plot_base_dir,
            fit_name=fit_name,
            transform_name=transform_name,
            dated=include_date
        )
        fname = f"rotation_{region}.mp4"
        return os.path.join(base, fname)

'''
