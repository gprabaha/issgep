import logging
from pathlib import Path

from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fixation_config import FixationConfig
from socialgaze.config.interactivity_config import InteractivityConfig

from socialgaze.data.gaze_data import GazeData
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityPlotter

logger = logging.getLogger(__name__)


def main():
    # === Configs & core data ===
    base_config = BaseConfig()
    fixation_config = FixationConfig()
    interactivity_config = InteractivityConfig()

    # All Figure 2 exports under BaseConfig.plots_dir / "figure_2"
    fig2_dir = Path(base_config.plots_dir) / "figure_2"
    fig2_dir.mkdir(parents=True, exist_ok=True)

    gaze_data = GazeData(config=base_config)
    fixdet = FixationDetector(gaze_data=gaze_data, config=fixation_config)
    plotter = InteractivityPlotter(config=interactivity_config)

    # =============================
    # Panel A: Face–Face previews
    # =============================
    # plotter.preview_random_runs(
    #     fixation_detector=fixdet,
    #     pairing=("m1", "face", "m2", "face"),
    #     n_samples=2,
    #     seed=None  # re-run to get different samples
    # )

    # # Pick 2 representative runs (edit to your choices)
    # reps_face_face = [("01072019", 6), ("02062018", 8), ("01112019", 9)]
    # plotter.export_representative_runs(
    #     fixation_detector=fixdet,
    #     pairing=("m1", "face", "m2", "face"),
    #     session_runs=reps_face_face,
    #     export_dir=fig2_dir / "face_face"
    # )

    # # =============================
    # # Panel C (control): Obj–Face previews
    # # =============================
    # plotter.preview_random_runs(
    #     fixation_detector=fixdet,
    #     pairing=("m1", "obj", "m2", "face"),
    #     n_samples=8,
    #     seed=None
    # )

    # reps_obj_face = [("02062018", 8), ("09042018", 3)]
    # plotter.export_representative_runs(
    #     fixation_detector=fixdet,
    #     pairing=("m1", "obj", "m2", "face"),
    #     session_runs=reps_obj_face,
    #     export_dir=fig2_dir / "obj_face"
    # )

    # =============================
    # Panel D/E: Face-fixation pies
    # =============================
    plotter.plot_face_fixation_pies_by_agent(
        fixation_detector=fixdet,
        export_dir=fig2_dir / "pies"
    )

    # logger.info("Figure 2 assets generated.")


if __name__ == "__main__":
    main()
