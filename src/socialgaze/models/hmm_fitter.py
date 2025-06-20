# src/socialgaze/models/hmm_fitter.py

import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

from hmmlearn.hmm import CategoricalHMM
from tqdm import tqdm


logger = logging.getLogger(__name__)

class HMMFitter:

    def __init__(self, config, fixation_detector, crosscorr_calculator, interactivity_detector):
        self.config = config
        self.fixation_detector = fixation_detector
        self.crosscorr_calculator = crosscorr_calculator
        self.interactivity_detector = interactivity_detector

        self.behavior_types = config.binary_vector_types_to_use
        self.output_dir = config.hmm_model_output_path


    def fit_hmm_all_runs(self):
        all_vector_data = self._collect_joint_categorical_vectors()

        logger.info("Fitting HMM to concatenated categorical sequence across all runs...")
        hmm_input_sequence = np.concatenate([seq for (_, _, seq) in all_vector_data])

        model = self._fit_hmm_to_sequence(hmm_input_sequence)

        save_path = self.output_dir / "hmm_model.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved HMM model to {save_path}")


    def _collect_joint_categorical_vectors(self) -> List[Tuple[str, int, np.ndarray]]:
        """
        Loads binary vector DataFrames from disk for each behavior type,
        constructs joint categorical sequences (m1 + m2 behaviors),
        and returns a list of (session, run, joint_vector).
        """
        all_dfs = []
        for btype in self.behavior_types:
            df = self.fixation_detector.get_binary_vector_df(behavior_type=btype)
            df["behavior_type"] = btype
            all_dfs.append(df)

        full_df = pd.concat(all_dfs, ignore_index=True)

        session_runs = full_df[["session_name", "run_number"]].drop_duplicates()
        results = []

        for _, row in tqdm(session_runs.iterrows(), total=len(session_runs), desc="Building joint behavior vectors"):
            session = row["session_name"]
            run = row["run_number"]

            run_df = full_df.query("session_name == @session and run_number == @run")

            m1_vector = self._encode_agent_behaviors(run_df, "m1")
            m2_vector = self._encode_agent_behaviors(run_df, "m2")
            joint_vector = self._combine_agent_vectors(m1_vector, m2_vector)

            results.append((session, run, joint_vector))

        return results


    def _encode_agent_behaviors(self, df: pd.DataFrame, agent: str) -> np.ndarray:
        """
        Converts multiple binary vectors for one agent into a categorical vector.
        0 = no active behavior; 1 = face_fixation; 2 = saccade_to_face; 3 = saccade_from_face.
        If multiple behaviors are active, the one with the highest priority in `self.behavior_types` is used.
        """
        run_subset = df[df["agent"] == agent]
        run_subset = run_subset.groupby("behavior_type")["binary_vector"].first().to_dict()

        # All vectors must be the same length
        T = len(next(iter(run_subset.values())))
        category_vector = np.zeros(T, dtype=int)

        for i, btype in enumerate(self.behavior_types):
            if btype in run_subset:
                vec = np.array(run_subset[btype])
                category_vector[vec == 1] = i + 1  # Offset by 1 to reserve 0 for 'none'

        return category_vector


    def _combine_agent_vectors(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Combines two categorical vectors into a single joint categorical vector
        with encoding: joint_code = v1 * N + v2
        """
        n = len(self.behavior_types) + 1
        return v1 * n + v2


    def _fit_hmm_to_sequence(self, seq: np.ndarray):
        n_obs = (len(self.behavior_types) + 1) ** 2
        model = CategoricalHMM(n_components=self.config.num_states, n_iter=100, verbose=True)
        model.fit(seq.reshape(-1, 1))
        return model


    def decode_sequence(self, model, sequence: np.ndarray) -> np.ndarray:
        """Decodes the latent states from an observed categorical sequence."""
        logprob, states = model.decode(sequence.reshape(-1, 1), algorithm="viterbi")
        return states


    def inverse_transform(self, joint_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts joint categorical vector back to two agent vectors.
        """
        n = len(self.behavior_types) + 1
        v1 = joint_vector // n
        v2 = joint_vector % n
        return v1, v2
