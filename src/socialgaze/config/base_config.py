# src/socialgaze/config/base_config.py

import pdb
import os
from pathlib import Path
import logging
import socket
import getpass
from typing import Dict, Optional, List
import pandas as pd

from socialgaze.utils.loading_utils import load_df_from_pkl, load_config_from_json
from socialgaze.utils.path_utils import (
    determine_root_data_dir,
    get_project_root,
    get_default_config_folder,
    get_ephys_days_df_pkl_path,
    get_position_df_pkl_path,
    get_pupil_df_pkl_path,
    get_roi_df_pkl_path,
    get_time_df_pkl_path,
    get_run_lengths_df_pkl_path,
    get_spike_times_mat_path,
    get_spike_df_pkl_path,
    get_default_data_paths,
    get_raw_data_directories,
    get_pupil_file_path,
    get_roi_file_path,
    get_time_file_path,
    join_folder_and_filename
)
from socialgaze.utils.discovery_utils import (
    get_config_filename,
    get_mat_filename_pattern,
    find_valid_sessions,
    filter_sessions_with_ephys,
    get_num_available_cpus
)
from socialgaze.utils.saving_utils import save_config_to_json
from socialgaze.utils.conversion_utils import object_to_dict, assign_dict_attributes_to_object


logger = logging.getLogger(__name__)


class BaseConfig:
    """
    A configuration manager for handling paths, environment flags,
    and available session/run combinations.

    Responsibilities:
    - Detect runtime environment (cluster/local)
    - Set default paths (raw, processed, outputs, config)
    - Determine valid sessions and runs
    - Save/load config from a JSON file
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the BaseConfig object.

        If a saved config exists at the provided path, it loads it.
        Otherwise, it auto-detects the environment, sets up paths, and scans for valid sessions/runs.
        On non-cluster environments, raw data paths and scanning are skipped.

        Args:
            config_path (Optional[str]): Optional path to an existing saved config file.
        """
        self.detect_runtime_environment()
        self.project_root = get_project_root()
        self.config_folder = get_default_config_folder(self.project_root)
        self.filename = get_config_filename(self.is_cluster, self.is_grace)
        
        # Determine config save path
        self.config_path = Path(config_path) if config_path else join_folder_and_filename(self.config_folder, self.filename)

        # Core processed output paths (these are safe to set in any environment)
        paths = get_default_data_paths(self.project_root)
        self.processed_data_dir = paths["processed"]
        self.output_dir = paths["outputs"]
        self.plots_dir = paths["plots"]

        # Set all processed dataframe paths via path_utils
        self.ephys_days_and_monkeys_df_path = get_ephys_days_df_pkl_path(self)
        self.positions_df_path = get_position_df_pkl_path(self)
        self.pupil_df_path = get_pupil_df_pkl_path(self)
        self.roi_vertices_df_path = get_roi_df_pkl_path(self)
        self.neural_timeline_df_path = get_time_df_pkl_path(self)
        self.run_length_df_path = get_run_lengths_df_pkl_path(self)
        self.spiketimes_df_path = get_spike_df_pkl_path(self)

        # Get days/sessions with ephys data and the corresponding monkeys
        self.ephys_days_and_monkeys_df = load_df_from_pkl(self.ephys_days_and_monkeys_df_path)
        self.monkey_dominance_df = None
        self.create_monkey_dominance_df()


        self.num_cpus = get_num_available_cpus(self.is_cluster)

        self.behav_data_types = ['positions', 'roi_vertices', 'pupil', 'neural_timeline']
        self.session_names: List[str] = []
        self.runs_by_session: Dict[str, List[str]] = {}

        if self.is_cluster:
            # Cluster: Discover from filesystem
            self.data_dir = determine_root_data_dir(
                is_cluster=self.is_cluster,
                is_grace=self.is_grace,
                prabaha_local=self.prabaha_local
            )
            raw_dirs = get_raw_data_directories(self.data_dir)
            self.position_dir = raw_dirs["position"]
            self.time_dir = raw_dirs["time"]
            self.pupil_dir = raw_dirs["pupil"]
            self.roi_dir = raw_dirs["roi"]

            self.file_pattern = get_mat_filename_pattern()
            self.initialize_sessions_and_runs()

            self.spiketimes_mat_path = get_spike_times_mat_path(self)
            self.extract_sessions_with_ephys_data(self.ephys_days_and_monkeys_df)
        else:
            # Local machine: load cached session/run metadata if available
            cache_path = self.processed_data_dir / "discovered_sessions_and_runs.pkl"
            if cache_path.exists():
                df = pd.read_pickle(cache_path)
                self.session_names = sorted(df["session_name"].unique().tolist())
                self.runs_by_session = (
                    df.groupby("session_name")["run_number"]
                    .apply(lambda x: sorted(x.astype(str).unique().tolist()))
                    .to_dict()
                )
                logger.info(f"Loaded session/run metadata from {cache_path}")
            else:
                logger.warning("Session/run metadata not available locally. Please initialize on the cluster first.")




    # -----------------------------
    # Config environment
    # -----------------------------

    def detect_runtime_environment(self) -> None:
        """
        Auto-detects the machine/runtime environment and sets flags:
        - is_cluster: whether running on HPC
        - is_grace: whether running on Grace cluster
        - prabaha_local: whether running locally on Prabaha's machine
        """
        hostname = socket.gethostname().lower()
        username = getpass.getuser()

        if "grace" in hostname or os.path.exists("/gpfs/gibbs/"):
            self.is_cluster = True
            self.is_grace = True
            self.prabaha_local = False
        elif "milgram" in hostname or os.path.exists("/gpfs/milgram/"):
            self.is_cluster = True
            self.is_grace = False
            self.prabaha_local = False
        elif username == "prabaha":
            self.is_cluster = False
            self.is_grace = False
            self.prabaha_local = True
        else:
            self.is_cluster = False
            self.is_grace = False
            self.prabaha_local = False

    # -----------------------------
    # Session / run discovery
    # -----------------------------

    def initialize_sessions_and_runs(self) -> None:
        """
        Populates self.session_names and self.runs_by_session using validity checks.
        Saves the result to disk for offline use.
        """
        self.session_names, self.runs_by_session = find_valid_sessions(
            self,
            path_fns={
                'time': get_time_file_path,
                'pupil': get_pupil_file_path,
                'roi': get_roi_file_path,
            }
        )

        # Save to disk
        session_df = pd.DataFrame([
            {"session_name": s, "run_number": run}
            for s in self.session_names
            for run in self.runs_by_session[s]
        ])
        session_df.to_pickle(self.processed_data_dir / "discovered_sessions_and_runs.pkl")


    def create_monkey_dominance_df(self):
        data = {
            'Monkey Pair': [
                'Kuro vs Ephron',
                'Kuro vs Lynch',
                'Kuro vs Hitchcock',
                'Lynch vs Cronenberg',
                'Lynch vs Ephron',
                'Lynch vs Hitchcock'
            ],
            'Olga': ['Kuro', 'Kuro', 'Hitchcock', 'Cronenberg', 'Lynch', 'Hitchcock'],
            'Siqi': ['Kuro', 'Kuro', 'Hitchcock', 'Cronenberg', 'Lynch', 'Hitchcock'],
            'Amrita': ['Kuro', 'Lynch', 'Hitchcock', 'Lynch', 'Lynch', 'Lynch']
        }

        df = pd.DataFrame(data)
        df[['m1', 'm2']] = df['Monkey Pair'].str.split(' vs ', expand=True)

        df['dominant_name'] = df.apply(_get_majority_dominant, axis=1)
        df['dominant_agent_label'] = df.apply(_get_agent_label, axis=1)

        df = df[['Monkey Pair', 'm1', 'm2', 'Olga', 'Siqi', 'Amrita',
                 'dominant_name', 'dominant_agent_label']]

        self.monkey_dominance_df = df
        df.to_pickle(self.processed_data_dir / "dominant_monkey_consensus.pkl")


    def extract_sessions_with_ephys_data(self, ephys_days_and_monkeys_df: pd.DataFrame) -> None:
        """
        Filters session/run list to only include those with matching ephys data.

        Args:
            ephys_days_and_monkeys_df (pd.DataFrame): Contains ephys recording metadata.
        """
        self.session_names, self.runs_by_session = filter_sessions_with_ephys(
            self.session_names,
            self.runs_by_session,
            ephys_days_and_monkeys_df
        )
        # Save to disk
        session_df = pd.DataFrame([
            {"session_name": s, "run_number": run}
            for s in self.session_names
            for run in self.runs_by_session[s]
        ])
        session_df.to_pickle(self.processed_data_dir / "discovered_sessions_and_runs.pkl")

    # -----------------------------
    # Save / load
    # -----------------------------

    def save_to_file(self, config_path: str) -> None:
        """
        Serializes the config as a dictionary and saves to a JSON file.

        Args:
            config_path (str): Path where the config should be saved.
        """
        save_config_to_json(object_to_dict(self), config_path)

    def load_from_file(self, config_path: str) -> None:
        """
        Loads config values from a JSON file and sets them as instance attributes.

        Args:
            config_path (str): Path to the saved config file.
        """
        config_data = load_config_from_json(config_path)
        assign_dict_attributes_to_object(self, config_data)

####################
# Helper functions #
####################

def _get_majority_dominant(row):
    """Returns the monkey with 2/3 agreement, or None if no agreement."""
    votes = [row['Olga'], row['Siqi'], row['Amrita']]
    for monkey in set(votes):
        if votes.count(monkey) >= 2:
            return monkey
    return None

def _get_agent_label(row):
    """Returns 'm1' if dominant is m1, 'm2' if dominant is m2, else None."""
    if pd.isna(row['dominant_name']):
        return None
    elif row['dominant_name'] in row['m1']:
        return 'm1'
    elif row['dominant_name'] in row['m2']:
        return 'm2'
    else:
        return None