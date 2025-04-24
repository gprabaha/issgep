# src/socialgaze/config/neural_config.py

from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.path_utils import get_psth_output_path, get_spike_data_dir


class NeuralConfig(BaseConfig):
    def __init__(self, config_path: str = None):
        """
        Configuration for neural data analysis, including PSTH extraction parameters.
        
        Args:
            config_path (str, optional): Path to a JSON config file. If None, default values are used.
        """
        super().__init__(config_path)

        # === PSTH Extraction Parameters ===
        self.use_parallel = True
        self.do_smoothing = True
        self.smoothing_bin_sigma = 2
        self.psth_bin_size = 0.01  # in seconds
        self.psth_window = (-0.5, 1)  # window around event (s before, s after)
        
        # === Data Paths ===
        self.
        self.psth_per_trial_path = get_psth_per_trial_path(self)  # Where to save the PSTH dataframe
