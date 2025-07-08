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
        self.mean_center_for_angle = True  # or False

        # Paths manager
        self.pca_paths = PCAPaths(self)

        # Final output plot dirs
        self.trajectories_dir = self.pca_paths.get_trajectories_dir()
        self.evr_bars_dir = self.pca_paths.get_evr_bars_dir()
