# scripts/misc/plot_inter_run_interval.py

import pdb
import os
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from socialgaze.config.base_config import BaseConfig
from socialgaze.data.gaze_data import GazeData

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def count_nan_padding(timeline_array):
    """
    Counts how many NaNs are at the beginning and end of the array.
    Assumes 1 kHz sampling rate (1 ms per sample).
    """
    timeline_array = np.asarray(timeline_array)

    # Count NaNs at start
    start_nan_count = 0
    for val in timeline_array:
        if np.isnan(val):
            start_nan_count += 1
        else:
            break

    # Count NaNs at end
    end_nan_count = 0
    for val in reversed(timeline_array):
        if np.isnan(val):
            end_nan_count += 1
        else:
            break

    return start_nan_count, end_nan_count


def main():
    # Setup
    config = BaseConfig()
    gaze_data = GazeData(config)

    logger.info("Loading raw neural timeline from .mat files...")
    gaze_data.load_raw_data_from_mat_files(data_types=["neural_timeline"])

    df = gaze_data.raw_data.get("neural_timeline")

    if df is None or df.empty:
        logger.warning("No raw neural timeline data found.")
        return
    
    # Compute inter-run intervals
    logger.info("Computing inter-run intervals...")
    inter_run_deltas = []

    grouped = df.groupby("session_name")

    for session_name, session_df in grouped:
        session_df["run_number"] = session_df["run_number"].astype(int)
        session_df = session_df.sort_values(by="run_number")
        timelines = session_df["neural_timeline"].tolist()
        
        for i in range(1, len(timelines)):
            prev_full = np.asarray(timelines[i - 1])
            curr_full = np.asarray(timelines[i])

            prev_start_nan, prev_end_nan = count_nan_padding(prev_full)
            curr_start_nan, curr_end_nan = count_nan_padding(curr_full)
            
            # Get first and last non-NaN values
            prev_valid = prev_full[~np.isnan(prev_full)]
            curr_valid = curr_full[~np.isnan(curr_full)]
            
            if len(prev_valid) == 0 or len(curr_valid) == 0:
                continue  # skip if all NaNs

            true_prev_end = prev_valid[-1] + (prev_end_nan * 0.001) # add padding to the end (in sec)
            true_curr_start = curr_valid[0] - (curr_start_nan * 0.001)  # subtract padding from the start (in sec)
            
            delta = true_curr_start - true_prev_end  # in s
            if delta > 0:
                inter_run_deltas.append(delta / 60)  # convert s to minutes

    if not inter_run_deltas:
        logger.warning("No inter-run intervals found.")
        return

    # Plot histogram
    date_str = datetime.today().strftime('%Y-%m-%d')
    out_dir = config.plots_dir / "inter_run_interval" / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inter_run_interval_histogram.png"

    bins = np.arange(1, 25)  # 1 to 15 minutes

    plt.figure(figsize=(8, 6))
    plt.hist(inter_run_deltas, bins=bins, color='skyblue', edgecolor='black', align='left')

    plt.title("Inter-run Interval Histogram")
    plt.xlabel("Time between runs (minutes)")
    plt.ylabel("Frequency")
    plt.xticks(bins)
    plt.grid(True)
    plt.tight_layout()

    if len(bins) > 10:
        plt.xticks(rotation=45)

    plt.savefig(out_path)
    logger.info(f"Histogram saved to {out_path}")


if __name__ == "__main__":
    main()
