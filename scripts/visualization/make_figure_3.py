
import logging
from pathlib import Path

from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.psth_config import PSTHConfig
from socialgaze.config.interactivity_config import InteractivityConfig

from socialgaze.data.gaze_data import GazeData
from socialgaze.data.spike_data import SpikeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityPlotter
from socialgaze.features.psth_extractor import PSTHExtractor

logger = logging.getLogger(__name__)


def main():
    # === Configs & core data ===
    base_config = BaseConfig()
    fixation_config = FixationConfig()
    psth_config = PSTHConfig()
    interactivity_config = InteractivityConfig()

    # All Figure 2 exports under BaseConfig.plots_dir / "figure_2"
    fig2_dir = Path(base_config.plots_dir) / "figure_2"
    fig2_dir.mkdir(parents=True, exist_ok=True)

    gaze_data = GazeData(config=base_config)
    spike_data = SpikeData(config=base_config)
    fixdet = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    plotter = InteractivityPlotter(config=interactivity_config)

    extractor = PSTHExtractor(
        config=psth_config,
        gaze_data=gaze_data,
        spike_data=spike_data,
        fixation_detector=fixation_detector,
        interactivity_detector=interactivity_detector,
    )

