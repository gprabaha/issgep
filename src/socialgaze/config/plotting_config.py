
import os
from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.path_utils import get_pc_plot_path

class PlottingConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.pc_plot_base_dir = "outputs/plots/pc_projection"
        self.plot_file_format = "png"
        self.plot_size = (8, 6)
        self.plot_dpi = 100
        self.include_date = True

    def get_static_plot_path(self, fit_name: str, transform_name: str, region: str) -> str:
        base = get_pc_plot_path(
            base_dir=self.pc_plot_base_dir,
            fit_name=fit_name,
            transform_name=transform_name,
            dated=self.include_date
        )
        fname = f"all_conditions_{region}.{self.plot_file_format}"
        return os.path.join(base, fname)

    def get_rotation_plot_path(self, fit_name: str, transform_name: str, region: str) -> str:
        base = get_pc_plot_path(
            base_dir=self.pc_plot_base_dir,
            fit_name=fit_name,
            transform_name=transform_name,
            dated=self.include_date
        )
        fname = f"rotation_{region}.mp4"
        return os.path.join(base, fname)
