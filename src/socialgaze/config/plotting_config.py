# src/socialgaze/config/plotting_config.py

from socialgaze.config.base_config import BaseConfig

class PlottingConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.plot_file_format = "png"
        self.plot_size = (8, 6)
        self.plot_dpi = 100
