from socialgaze.config.base_config import BaseConfig

class PlottingConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.pc_plot_base_dir = "outputs/plots/pc_projection"
        self.plot_file_format = "png"
        self.plot_dpi = 300
