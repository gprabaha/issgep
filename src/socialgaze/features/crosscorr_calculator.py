# src/socialgaze/features/crosscorr_calculator.py

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm
from scipy.signal import fftconvolve
import logging
import os

from socialgaze.config.crosscorr_config import CrossCorrConfig
from socialgaze.features.fixation_detector import FixationDetector
from socialgaze.features.interactivity_detector import InteractivityDetector
from socialgaze.utils.path_utils import get_crosscorr_output_path
from socialgaze.utils.loading_utils import load_df_from_pkl
from socialgaze.utils.saving_utils import save_df_to_pkl

logger = logging.getLogger(__name__)


class CrossCorrCalculator:
    def __init__(
        self,
        config: CrossCorrConfig,
        fixation_detector: FixationDetector,
        interactivity_detector: Optional[InteractivityDetector] = None,
    ):
        self.config = config
        self.fixation_detector = fixation_detector
        self.interactivity_detector = interactivity_detector

    def compute_and_save_crosscorrelations(self):
        logger.info("Computing and saving inter-agent cross-correlations...")
        behavior_types = self.config.binary_vector_types_to_use
        all_dfs = []
        for btype in behavior_types:
            try:
                df = self.fixation_detector.get_binary_vector_df(btype)
                all_dfs.append(df)
            except FileNotFoundError:
                logger.warning(f"Missing binary vector for: {btype}")
        if not all_dfs:
            logger.warning("No binary vector data loaded. Exiting cross-correlation calculation.")
            return
        bv_df = pd.concat(all_dfs, ignore_index=True)
        grouped = bv_df.groupby(["session_name", "run_number"])
        agent_behaviors = bv_df[["agent", "behavior_type"]].drop_duplicates().values.tolist()
        for (a1, b1) in tqdm(agent_behaviors, desc="Source agent-behavior loop"):
            for (a2, b2) in agent_behaviors:
                if a1 == a2:
                    continue  # Only cross-agent

                comparison_name = f"{a1}_{b1}__vs__{a2}_{b2}"
                all_rows = []
                for (session, run), run_df in tqdm(grouped, desc=f"Runs for {comparison_name}", leave=False):
                    agent_data = {
                        (row["agent"], row["behavior_type"]): row["binary_vector"]
                        for _, row in run_df.iterrows()
                    }
                    if (a1, b1) not in agent_data or (a2, b2) not in agent_data:
                        continue

                    v1 = agent_data[(a1, b1)]
                    v2 = agent_data[(a2, b2)]
                    lags, corr = _compute_normalized_crosscorr(
                        v1, v2,
                        max_lag=self.config.max_lag,
                        normalize=self.config.normalize
                    )
                    rows = pd.DataFrame({
                        "session_name": session,
                        "run_number": run,
                        "agent1": a1,
                        "agent2": a2,
                        "behavior1": b1,
                        "behavior2": b2,
                        "lag": lags,
                        "value": corr
                    })
                    all_rows.append(rows)
                if all_rows:
                    full_df = pd.concat(all_rows, ignore_index=True)
                    save_path = get_crosscorr_output_path(self.config, comparison_name)
                    os.makedirs(save_path.parent, exist_ok=True)
                    save_df_to_pkl(full_df, save_path)
        logger.info("All cross-correlations saved.")


    def load_crosscorr_df(self, comparison_name: str) -> pd.DataFrame:
        path = get_crosscorr_output_path(self.config, comparison_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cross-correlation result not found at: {path}")
        return load_df_from_pkl(path)


def _compute_normalized_crosscorr(x: np.ndarray, y: np.ndarray, max_lag: int, normalize: bool = True):
    x = x.astype(float)
    y = y.astype(float)
    corr_full = fftconvolve(x, y[::-1], mode="full")
    center = len(corr_full) // 2
    lags = np.arange(-max_lag, max_lag + 1)
    corr = corr_full[center - max_lag:center + max_lag + 1]

    if normalize:
        x_std = np.std(x)
        y_std = np.std(y)
        if x_std > 0 and y_std > 0:
            corr /= (len(x) * x_std * y_std)
        else:
            corr[:] = 0
    return lags, corr
