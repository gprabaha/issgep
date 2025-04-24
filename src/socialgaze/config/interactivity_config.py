# src/socialgaze/config/interactivity_config.py

from socialgaze.config.fixation_config import FixationConfig
from socialgaze.utils.path_utils import get_mutual_fixation_density_path, get_interactivity_df_path


class InteractivityConfig(FixationConfig):
    def __init__(self, base_config: FixationConfig = None):
        super().__init__()
        if base_config:
            self.__dict__.update(base_config.__dict__)

        self.interactivity_threshold = 0.63
        self.use_parallel = True
        self.fixation_type_to_process = 'face'
        self.mutual_fixation_density_path = get_mutual_fixation_density_path(self, fixation_type=self.fixation_type_to_process)
        self.interactivity_df_path = get_interactivity_df_path(self)
