# scripts/preprocessing/load_all_raw_mat_and_save_as_df.py

import os
import sys
import subprocess
import logging
import pandas as pd
from config.base_config import BaseConfig
from src.data.extract_data_from_mat_files import (
    process_position_file,
    process_time_file,
    process_pupil_file,
    process_roi_rects_file,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config_path = "config/saved_configs/local_default.json"

    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at: {config_path}")
        logger.info("Attempting to generate config automatically...")
        try:
            subprocess.run(["python", "scripts/setup/make_config_file.py"], check=True)
            logger.info(f"Config generated at {config_path}")
        except subprocess.CalledProcessError:
            logger.error("Failed to generate config. Exiting.")
            sys.exit(1)

    config = BaseConfig(config_path=config_path)

    all_positions = []
    all_pupils = []
    all_timelines = []
    all_rois = []

    for session_date in config.session_dates:
        for run_number in config.runs_by_session.get(session_date, []):
            run_number = str(run_number)

            for agent in ["m1", "m2"]:
                pos_df = process_position_file(config.get_position_file_path(session_date, run_number),
                                               agent, session_date, run_number)
                if pos_df is not None:
                    all_positions.append(pos_df)

                pupil_df = process_pupil_file(config.get_pupil_file_path(session_date, run_number),
                                              agent, session_date, run_number)
                if pupil_df is not None:
                    all_pupils.append(pupil_df)

                roi_df = process_roi_rects_file(config.get_roi_file_path(session_date, run_number),
                                                agent, session_date, run_number)
                if roi_df is not None:
                    all_rois.append(roi_df)

            timeline_df = process_time_file(config.get_time_file_path(session_date, run_number),
                                            session_date, run_number)
            if timeline_df is not None:
                all_timelines.append(timeline_df)

    out_dir = config.processed_dir
    os.makedirs(out_dir, exist_ok=True)

    if all_positions:
        pd.concat(all_positions).to_pickle(out_dir / "positions.pkl")
        logger.info(f"Saved positions.pkl to {out_dir}")
    if all_pupils:
        pd.concat(all_pupils).to_pickle(out_dir / "pupil.pkl")
        logger.info(f"Saved pupil.pkl to {out_dir}")
    if all_timelines:
        pd.concat(all_timelines).to_pickle(out_dir / "neural_timeline.pkl")
        logger.info(f"Saved neural_timeline.pkl to {out_dir}")
    if all_rois:
        pd.concat(all_rois).to_pickle(out_dir / "roi_rects.pkl")
        logger.info(f"Saved roi_rects.pkl to {out_dir}")

    logger.info("All raw data loaded and saved as DataFrames.")


if __name__ == "__main__":
    main()
