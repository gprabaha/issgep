# scripts/visualization/make_figure_3.py

import logging
from pathlib import Path

from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.psth_config import PSTHConfig
from socialgaze.config.interactivity_config import InteractivityConfig

from socialgaze.data.gaze_data import GazeData
from socialgaze.data.spike_data import SpikeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.features.psth_extractor import PSTHPlotter  # plotter lives in psth_extractor.py

logger = logging.getLogger(__name__)


def main():
    # === Configs & core data ===
    base_config = BaseConfig()
    fixation_config = FixationConfig()
    psth_config = PSTHConfig()
    interactivity_config = InteractivityConfig()

    # Figure 3 exports go under BaseConfig.plots_dir / "figure_3"
    fig3_dir = Path(base_config.plots_dir) / "figure_3"
    fig3_dir.mkdir(parents=True, exist_ok=True)

    # Optional: set detection thresholds here (fallbacks exist: 5 & 25)
    if not hasattr(psth_config, "min_consecutive_sig_bins"):
        psth_config.min_consecutive_sig_bins = 5
    if not hasattr(psth_config, "min_total_sig_bins"):
        psth_config.min_total_sig_bins = 25

    # === Data objects ===
    gaze_data = GazeData(config=base_config)
    spike_data = SpikeData(config=base_config)
    fixation_detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    interactivity_detector = InteractivityDetector(config=interactivity_config)

    # === PSTH analysis & plotting ===
    plotter = PSTHPlotter(
        config=psth_config,
        gaze_data=gaze_data,
        spike_data=spike_data,
        fixation_detector=fixation_detector,
        interactivity_detector=interactivity_detector,
    )

    # 1) Compute & cache significance table (stored at: psth_config.output_dir / "results" / interactive_face_significance.pkl)
    sig_df = plotter.compute_interactive_face_significance()
    if sig_df is None or sig_df.empty:
        logger.warning("No significant units table produced; stopping before plotting.")
        return

    # 2) Make Illustrator‑friendly unit plots + per‑region pies
    #    Output dir: base_config.plots_dir / "psth" / "interactive_units" / <REGION> / ...
    # plotter.plot_significant_interactive_vs_noninteractive_units()

    # logger.info(f"Figure 3 assets written under:\n  - {fig3_dir}\n  - {Path(base_config.plots_dir) / 'psth' / 'interactive_units'}")

    plotter.plot_region_heatmaps_of_sig_units()

    plotter.plot_region_violin_summaries()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
