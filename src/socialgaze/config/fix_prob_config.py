# src/socialgaze/config/fix_prob_config.py

from socialgaze.config.fixation_config import FixationConfig
from socialgaze.utils.path_utils import (
    get_fixation_probability_path,
    get_fixation_probability_by_interactivity_path,
    get_fixation_probability_by_interactivity_segment_path,
    get_fixation_probability_plot_dir,
)

class FixProbConfig(FixationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fix_prob_df_path = get_fixation_probability_path(self)
        self.fix_prob_df_by_interactivity_path = get_fixation_probability_by_interactivity_path(self)
        self.fix_prob_df_by_interactivity_segment_path = get_fixation_probability_by_interactivity_segment_path(self)

        self.plot_dir = get_fixation_probability_plot_dir(self)
        self.violin_palette = {"P(m1)*P(m2)": "#66c2a5", "P(m1&m2)": "#8da0cb"}