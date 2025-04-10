# scripts/preprocessing/prune_gaze_data_dfs.py


import logging
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from socialgaze.config.base_config import BaseConfig
from socialgaze.utils.config_utils import ensure_config_exists
from socialgaze.utils.loading_utils import load_df_from_pkl
from socialgaze.utils.saving_utils import save_df_to_pkl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def interpolate_nans(array, kind='linear', window_size=10, max_nans=3):
    if array.ndim == 1 or kind == 'linear':
        mask = np.isnan(array)
        if np.any(mask) and np.any(~mask):
            logger.debug(f"Interpolating {np.sum(mask)} NaNs with linear method.")
            array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), array[~mask])
        return array

    elif array.ndim == 2 and kind == 'sliding':
        num_points, num_dims = array.shape
        stride = max_nans
        global_nan_mask = np.isnan(array).any(axis=1)
        logger.debug(f"Interpolating NaNs in 2D positions with sliding window.")

        for start in range(0, num_points - window_size + 1, stride):
            end = start + window_size
            window_mask = global_nan_mask[start:end]
            nan_count = np.sum(window_mask)

            if 0 < nan_count <= max_nans:
                window = array[start:end].copy()
                for col in range(num_dims):
                    col_vals = window[:, col]
                    valid = np.where(~np.isnan(col_vals))[0]
                    if len(valid) > 1:
                        interp_func = interp1d(valid, col_vals[valid], kind='cubic', fill_value="extrapolate", bounds_error=False)
                        to_fill = np.where(window_mask)[0]
                        col_vals[to_fill] = interp_func(to_fill)
                array[start:end] = window
        return array

    else:
        raise ValueError("Unsupported interpolation type or array shape.")


def _get_row(df, agent):
    try:
        return df[df['agent'] == agent].iloc[0]
    except IndexError:
        return None


def prune_and_interpolate_all(df_timeline, df_positions, df_pupils):
    cleaned_pos_rows = []
    cleaned_pupil_rows = []
    cleaned_timeline_rows = []

    grouped = df_timeline.groupby(['session_name', 'run_number'])
    total_groups = len(grouped)
    logger.info(f"Processing {total_groups} session-run groups...")

    for i, ((session, run), time_group) in enumerate(grouped, 1):
        idx = time_group.index[0]
        timeline = time_group.loc[idx, 'neural_timeline']

        if timeline is None or len(timeline) == 0:
            logger.debug(f"[{session}, run {run}] Skipping empty timeline.")
            continue

        timeline = np.array(timeline)
        valid_idx = np.where(~np.isnan(timeline))[0]
        if len(valid_idx) == 0:
            logger.debug(f"[{session}, run {run}] Skipping timeline with all NaNs.")
            continue

        logger.debug(f"[{session}, run {run}] Valid indices found: {len(valid_idx)}")
        cleaned_timeline_rows.append(time_group.assign(neural_timeline=[timeline[valid_idx]]))

        pos_subset = df_positions.query("session_name == @session and run_number == @run")
        pupil_subset = df_pupils.query("session_name == @session and run_number == @run")

        for agent in ['m1', 'm2']:
            pos_row = _get_row(pos_subset, agent)
            pupil_row = _get_row(pupil_subset, agent)
            if pos_row is None or pupil_row is None:
                logger.debug(f"[{session}, run {run}, {agent}] Data missing. Skipping.")
                continue

            # Interpolate positions
            x = np.array(pos_row['x'])[valid_idx]
            y = np.array(pos_row['y'])[valid_idx]
            positions = np.stack([x, y], axis=1)
            positions = interpolate_nans(positions, kind='sliding', window_size=10, max_nans=3)
            pos_row['x'] = positions[:, 0]
            pos_row['y'] = positions[:, 1]
            cleaned_pos_rows.append(pos_row)

            # Interpolate pupils
            pupil = np.array(pupil_row['pupil_size'])[valid_idx]
            pupil = interpolate_nans(pupil, kind='linear')
            pupil_row['pupil_size'] = pupil
            cleaned_pupil_rows.append(pupil_row)

        if i % 50 == 0 or i == total_groups:
            logger.info(f"Processed {i}/{total_groups} groups...")

    return (
        pd.concat(cleaned_timeline_rows, ignore_index=True),
        pd.DataFrame(cleaned_pos_rows),
        pd.DataFrame(cleaned_pupil_rows),
    )


def main():
    config_path = "src/socialgaze/config/saved_configs/milgram_default.json"
    ensure_config_exists(config_path)
    config = BaseConfig(config_path=config_path)

    logger.info("Loading input dataframes...")
    df_timeline = load_df_from_pkl(config.processed_data_dir / "neural_timeline.pkl")
    df_positions = load_df_from_pkl(config.processed_data_dir / "positions.pkl")
    df_pupils = load_df_from_pkl(config.processed_data_dir / "pupil.pkl")

    logger.info("Pruning and interpolating gaze and pupil data...")
    cleaned_timeline_df, cleaned_positions_df, cleaned_pupils_df = prune_and_interpolate_all(
        df_timeline, df_positions, df_pupils
    )

    logger.info("Saving cleaned dataframes...")
    save_df_to_pkl(cleaned_timeline_df, config.processed_data_dir / "neural_timeline.pkl")
    save_df_to_pkl(cleaned_positions_df, config.processed_data_dir / "positions.pkl")
    save_df_to_pkl(cleaned_pupils_df, config.processed_data_dir / "pupil.pkl")

    logger.info("Done. Pruned and saved cleaned dataframes.")


if __name__ == "__main__":
    main()

