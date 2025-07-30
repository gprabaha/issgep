# src/socialgaze/config/foraging_config.py

from socialgaze.config.base_config import BaseConfig

class ForagingConfig(BaseConfig):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)

        self.use_parallel: bool = True
        self.show_inner_tqdm: bool = True
        self.include_interactivity: bool = True
        self.fixation_categories_to_use: list = [
            "face", "object", "out_of_roi"
        ]
