# src/socialgaze/config/fix_prob_config.py

from socialgaze.config.fixation_config import FixationConfig
from socialgaze.utils.path_utils import get_fixation_probability_path

class FixProbConfig(FixationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fix_prob_df_path = get_fixation_probability_path(self)
