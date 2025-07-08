# src/socialgaze/config/psth_config.py

from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.path_utils import PSTHPaths


class PSTHConfig(BaseConfig):
    """
    Configuration for PSTH extraction and saving.
    Includes:
      - Parameters for PSTH calculation
      - Centralized path manager for outputs and plots
    """

    def __init__(self, config_path: str = None):
        super().__init__(config_path)

        # === PSTH extraction parameters ===
        self.use_parallel = True
        self.do_smoothing = True
        self.smoothing_bin_sigma = 2  # Smoothing sigma in number of bins
        self.psth_bin_size = 0.01     # Bin size in seconds
        self.psth_window = (-0.5, 1)  # Window relative to event: (before, after) in seconds

        # === PSTH path manager ===
        self.psth_paths = PSTHPaths(self)

        # === Processed PSTH output paths ===
        self.psth_per_trial_path = self.psth_paths.get_psth_per_trial_path()
        self.avg_face_obj_path = self.psth_paths.get_avg_face_obj_path()
        self.avg_int_non_int_face_path = self.psth_paths.get_avg_int_non_int_face_path()

        # === PSTH plots output (date-specific) ===
        self.psth_plot_date_dir = self.psth_paths.get_psth_plot_date_dir()
