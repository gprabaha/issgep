# src/socialgaze/models/hmm_fitter.py

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
from pyhsmm.models import HSMM
from pyhsmm.basic.distributions import Categorical, PoissonDuration

logger = logging.getLogger(__name__)

class HMMFitter:
    def __init__(self, config, fixation_detector, crosscorr_calculator, interactivity_detector):
        self.config = config
        self.fixation_detector = fixation_detector
        self.crosscorr_calculator = crosscorr_calculator
        self.interactivity_detector = interactivity_detector

        self.behavior_types = config.binary_vector_types_to_use
        self.output_dir = config.hmm_model_output_path

    def fit_downsampled_hmm_all_runs(self, downsample_factor=50):
        all_vector_data = self._collect_joint_categorical_vectors(downsample_factor=downsample_factor)

        logger.info(f"Fitting HMM to downsampled sequences at 1/{downsample_factor} resolution...")
        sequences = [seq for (_, _, seq) in all_vector_data]
        lengths = [len(seq) for seq in sequences]
        stacked = np.concatenate(sequences).reshape(-1, 1)

        model = self._fit_hmm_to_sequence(stacked, lengths)

        save_path = self.output_dir / f"hmm_model_downsampled_{downsample_factor}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved HMM model to {save_path}")

    def fit_event_level_hsmm(self):
        all_obs_and_durs = self._collect_event_level_observations_and_durations()
        observations = [np.array(obs) for obs, _ in all_obs_and_durs]
        durations = [np.array(durs) for _, durs in all_obs_and_durs]

        logger.info("Fitting HSMM to behavioral event sequences with durations using pyhsmm...")

        K = self.config.num_states
        n_obs = (len(self.behavior_types) + 1) ** 2

        obs_distns = [Categorical(K=n_obs) for _ in range(K)]
        dur_distns = [PoissonDuration(alpha_0=2.0, beta_0=2.0) for _ in range(K)]

        model = HSMM(alpha=6., init_state_concentration=6., obs_distns=obs_distns, dur_distns=dur_distns)

        for obs, dur in zip(observations, durations):
            model.add_data(obs, lengths=dur)

        for idx in range(100):
            model.resample_model()

        save_path = self.output_dir / "hsmm_model_events_pyhsmm.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved HSMM model to {save_path}")

    def _collect_joint_categorical_vectors(self, downsample_factor=None) -> List[Tuple[str, int, np.ndarray]]:
        all_dfs = []
        for btype in self.behavior_types:
            df = self.fixation_detector.get_binary_vector_df(behavior_type=btype)
            df["behavior_type"] = btype
            all_dfs.append(df)

        full_df = pd.concat(all_dfs, ignore_index=True)
        session_runs = full_df[["session_name", "run_number"]].drop_duplicates()
        results = []

        for _, row in tqdm(session_runs.iterrows(), total=len(session_runs), desc="Building categorical vectors"):
            session, run = row["session_name"], row["run_number"]
            run_df = full_df.query("session_name == @session and run_number == @run")

            m1_vec = self._encode_agent_behaviors(run_df, "m1", downsample_factor)
            m2_vec = self._encode_agent_behaviors(run_df, "m2", downsample_factor)
            joint_vec = self._combine_agent_vectors(m1_vec, m2_vec)

            results.append((session, run, joint_vec))
        return results

    def _encode_agent_behaviors(self, df: pd.DataFrame, agent: str, downsample_factor=None) -> np.ndarray:
        sub = df[df["agent"] == agent]
        sub = sub.groupby("behavior_type")["binary_vector"].first().to_dict()
        T = len(next(iter(sub.values())))
        category_vector = np.zeros(T, dtype=int)

        for i, btype in enumerate(self.behavior_types):
            if btype in sub:
                vec = np.array(sub[btype])
                category_vector[vec == 1] = i + 1

        if downsample_factor:
            pad_len = len(category_vector) % downsample_factor
            if pad_len:
                category_vector = category_vector[:-pad_len]
            reshaped = category_vector.reshape(-1, downsample_factor)
            downsampled = np.apply_along_axis(lambda x: np.max(x), 1, reshaped)
            return downsampled

        return category_vector

    def _combine_agent_vectors(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        n = len(self.behavior_types) + 1
        return v1 * n + v2

    def _fit_hmm_to_sequence(self, seq: np.ndarray, lengths: List[int]):
        from hmmlearn.hmm import CategoricalHMM

        model = CategoricalHMM(n_components=self.config.num_states, n_iter=100, verbose=True)
        model.fit(seq, lengths)
        return model

    def _collect_event_level_observations_and_durations(self) -> List[Tuple[List[int], List[int]]]:
        all_dfs = []
        for btype in self.behavior_types:
            df = self.fixation_detector.get_binary_vector_df(behavior_type=btype)
            df["behavior_type"] = btype
            all_dfs.append(df)

        full_df = pd.concat(all_dfs, ignore_index=True)
        session_runs = full_df[["session_name", "run_number"]].drop_duplicates()
        all_obs_and_durs = []

        for _, row in session_runs.iterrows():
            session, run = row["session_name"], row["run_number"]
            run_df = full_df.query("session_name == @session and run_number == @run")

            m1_vec = self._encode_agent_behaviors(run_df, "m1")
            m2_vec = self._encode_agent_behaviors(run_df, "m2")
            joint_vec = self._combine_agent_vectors(m1_vec, m2_vec)

            obs, durs = self._compress_categorical_with_durations(joint_vec)
            all_obs_and_durs.append((obs, durs))

        return all_obs_and_durs

    def _compress_categorical_with_durations(self, vector: np.ndarray) -> Tuple[List[int], List[int]]:
        obs = []
        durs = []

        if len(vector) == 0:
            return obs, durs

        current = vector[0]
        count = 1
        for v in vector[1:]:
            if v == current:
                count += 1
            else:
                obs.append(current)
                durs.append(count)
                current = v
                count = 1
        obs.append(current)
        durs.append(count)
        return obs, durs


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
