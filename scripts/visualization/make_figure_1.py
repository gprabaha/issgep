import logging
from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.fix_prob_config import FixProbConfig
from socialgaze.config.interactivity_config import InteractivityConfig
from socialgaze.config.crosscorr_config import CrossCorrConfig

from socialgaze.data.gaze_data import GazeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.features.fix_prob_detector import FixProbDetector, FixProbPlotter
from socialgaze.features.crosscorr_calculator import CrossCorrCalculator

# Import the new FixationPlotter class (in same module as FixationDetector)
from socialgaze.features.fixation_detector import FixationPlotter, FaceFixPlotStyle

logger = logging.getLogger(__name__)

def main():
    # Load all configs
    base_config = BaseConfig()
    fixation_config = FixationConfig()
    fix_prob_config = FixProbConfig()
    interactivity_config = InteractivityConfig()
    crosscorr_config = CrossCorrConfig()

    # Initialize core data
    gaze_data = GazeData(config=base_config)

    # Initialize detectors (so detection is available if needed)
    fixation_detector = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    interactivity_detector = InteractivityDetector(config=interactivity_config)

    # # Initialize plotter (inherits from FixationDetector, uses same data/config)
    # fixation_plotter = FixationPlotter(gaze_data=gaze_data, config=fixation_config)
    # # === EXPORT SINGLE RUN PDFs ===
    # fixation_plotter.plot_face_fixation_timelines(
    #     export_pdf_for=("02062018", 8)
    # )
    # fixation_plotter.plot_face_fixation_timelines(
    #     export_pdf_for=("09042018", 3)
    # )

    fix_prob_plotter = FixProbPlotter(
        fixation_detector=fixation_detector,
        config=fix_prob_config,
        interactivity_detector=interactivity_detector
    )
    
    for mode in fix_prob_config.modes:
        print(f"\n=== Plotting fixation probability violins for mode = '{mode}' ===")
        # Plot all modes
        fix_prob_plotter.plot_joint_vs_marginal_violin(context=mode)

    # calculator = CrossCorrCalculator(
    #     config=crosscorr_config,
    #     fixation_detector=fixation_detector,
    #     interactivity_detector=interactivity_detector,
    # )

    # calculator.plot_crosscorr_deltas_leader_follower_all_full_facefix()


if __name__ == "__main__":
    main()
