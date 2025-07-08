# src/socialgaze/utils/path_utils.py

import pdb
import logging
import os
from pathlib import Path
from typing import Dict, Union
from datetime import date
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)
# --------------------
# == Root paths ==
# --------------------

def determine_root_data_dir(is_cluster: bool, is_grace: bool, prabaha_local: bool) -> str:
    """
    Determines the root directory for raw data based on the current runtime environment.

    Args:
        is_cluster (bool): Whether the code is running on a cluster.
        is_grace (bool): Whether the code is running on the Grace cluster.
        prabaha_local (bool): Whether the code is running on Prabaha's local machine.

    Returns:
        str: Path to the root raw data directory.
    """
    if is_cluster:
        return (
            Path("/gpfs/gibbs/project/chang/pg496/data_dir/social_gaze/")
            if is_grace else
            Path("/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/")
        )
    if prabaha_local:
        return Path("/Users/prabaha/data_dir/social_gaze")
    return Path("../data/raw")


def get_project_root() -> Path:
    """
    Returns the root directory of the project.

    This function assumes it is located 3 directories deep inside the project
    (e.g., inside src/socialgaze/utils/) and moves upward to find the root.

    Returns:
        Path: The root path of the entire project.
    """
    return Path(Path(__file__).resolve().parents[3])


def get_default_config_folder(project_root: Path) -> Path:
    """
    Returns the default configuration folder path and ensures it exists.

    This folder is typically used to store JSON config files generated during runtime.

    Args:
        project_root (Path): The root directory of the project.

    Returns:
        Path: Path to the config folder (e.g., src/socialgaze/config/saved_configs).
    """
    config_folder = project_root / "src/socialgaze/config/saved_configs"
    config_folder.mkdir(parents=True, exist_ok=True)
    return config_folder

# --------------------
# == Raw data paths ==
# --------------------

def get_default_data_paths(project_root: Path) -> Dict[str, Path]:
    """
    Constructs default paths to processed data, output, and plot folders.

    Args:
        project_root (Path): The root directory of the project.

    Returns:
        Dict[str, Path]: A dictionary with keys 'processed', 'outputs', and 'plots'.
    """
    return {
        "processed": project_root / "data/processed",
        "outputs": project_root / "outputs",
        "plots": project_root / "outputs/plots",
    }


def get_raw_data_directories(data_root: Path) -> Dict[str, Path]:
    """
    Returns standardized subdirectories for raw eyetracking data under the given root.

    Args:
        data_root (Path): The root directory for raw data (e.g., /gpfs/.../data)

    Returns:
        Dict[str, Path]: Dictionary containing paths for 'position', 'time', 'pupil', and 'roi'
    """
    return {
        "position": data_root / "eyetracking/aligned_raw_samples/position",
        "time": data_root / "eyetracking/aligned_raw_samples/time",
        "pupil": data_root / "eyetracking/aligned_raw_samples/pupil_size",
        "roi": data_root / "eyetracking/roi_rects",
    }


def get_position_file_path(config, session_date: str, run_number: str) -> Path:
    """
    Constructs the full path to the position .mat file for a given session and run.

    Args:
        config: The BaseConfig object containing path and filename info.
        session_date (str): The session date string.
        run_number (str): The run number string.

    Returns:
        Path: Full path to the position file.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return config.position_dir / filename


def get_time_file_path(config, session_date: str, run_number: str) -> Path:
    """
    Constructs the full path to the time .mat file for a given session and run.

    Args:
        config: The BaseConfig object containing path and filename info.
        session_date (str): The session date string.
        run_number (str): The run number string.

    Returns:
        Path: Full path to the time file.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return config.time_dir / filename


def get_pupil_file_path(config, session_date: str, run_number: str) -> Path:
    """
    Constructs the full path to the pupil .mat file for a given session and run.

    Args:
        config: The BaseConfig object containing path and filename info.
        session_date (str): The session date string.
        run_number (str): The run number string.

    Returns:
        Path: Full path to the pupil file.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return config.pupil_dir / filename


def get_roi_file_path(config, session_date: str, run_number: str) -> Path:
    """
    Constructs the full path to the ROI .mat file for a given session and run.

    Args:
        config: The BaseConfig object containing path and filename info.
        session_date (str): The session date string.
        run_number (str): The run number string.

    Returns:
        Path: Full path to the ROI file.
    """
    filename = config.file_pattern.format(session_date=session_date, run_number=run_number)
    return config.roi_dir / filename


def get_spike_times_mat_path(config) -> Path:
    """
    Returns the full path to the spike times .mat file.
    """
    return config.data_dir / "unit_spiketimes.mat"

# ---------------------------------------------------------
# == Processed behavioral and spike data df pickle paths ==
# ---------------------------------------------------------

def get_ephys_days_df_pkl_path(config) -> Path:
    return config.processed_data_dir / "ephys_days_and_monkeys.pkl"

def get_position_df_pkl_path(config) -> Path:
    return config.processed_data_dir / "positions.pkl"

def get_pupil_df_pkl_path(config) -> Path:
    return config.processed_data_dir / "pupil.pkl"

def get_roi_df_pkl_path(config) -> Path:
    return config.processed_data_dir / "roi_vertices.pkl"

def get_time_df_pkl_path(config) -> Path:
    return config.processed_data_dir / "neural_timeline.pkl"

def get_run_lengths_df_pkl_path(config) -> Path:
    return config.processed_data_dir / "run_lengths.pkl"

def get_fix_binary_vec_df_path(config) -> Path:
    return config.processed_data_dir / "fixation_binary_vectors.pkl"

def get_spike_df_pkl_path(config) -> Path:
    """
    Returns the path to the saved spike dataframe as .pkl.
    """
    return config.processed_data_dir / "spike_data.pkl"


# -----------------------------------------------------
# == Paths to data analysis result dataframe pickles ==
# -----------------------------------------------------

# == Fix and saccade paths ==

def get_fixation_df_path(processed_data_dir: Path) -> Path:
    """
    Returns the path to the final saved fixation dataframe.

    Args:
        processed_data_dir (Path): Base directory where processed data is stored.

    Returns:
        Path: Path to 'fixations.pkl'.
    """
    processed_data_dir = Path(processed_data_dir)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    return processed_data_dir / "fixations.pkl"


def get_saccade_df_path(processed_data_dir: Path) -> Path:
    """
    Returns the path to the final saved saccade dataframe.

    Args:
        processed_data_dir (Path): Base directory where processed data is stored.

    Returns:
        Path: Path to 'saccades.pkl'.
    """
    processed_data_dir = Path(processed_data_dir)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    return processed_data_dir / "saccades.pkl"


def get_behav_binary_vector_path(config, behavior_type: str) -> Path:
    """
    Returns the output path for a given binary vector type.
    e.g., face_fixation → processed_data_dir/binary_vectors/face_fixation.pkl
    """
    binary_vec_dir = Path(config.processed_data_dir) / "binary_vectors"
    binary_vec_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{behavior_type}.pkl"
    return binary_vec_dir / filename


# == Interactivity paths ==

def get_mutual_fixation_density_path(config, fixation_type='face') -> Path:
    return config.processed_data_dir / f"mutual_fixation_density_{fixation_type}.pkl"

def get_interactivity_df_path(config, fixation_type='face') -> Path:
    return config.processed_data_dir / "interactive_periods.pkl"


# == PSTH paths ==
 
def get_psth_per_trial_path(config) -> str:
    """
    Returns the path for saving/loading the per-trial PSTH dataframe.
    """
    return config.processed_data_dir / "psth_per_trial.pkl"


def get_avg_psth_per_category_path(config) -> str:
    """
    Returns the path for saving/loading the average PSTH per category dataframe.
    """
    return config.processed_data_dir / "avg_psth_per_category.pkl"


def get_avg_psth_per_category_and_interactivity_path(config) -> str:
    """
    Returns the path for saving/loading the average PSTH per category and interactivity dataframe.
    """
    return config.processed_data_dir / "avg_psth_per_category_and_interactivity.pkl"


# --------------------------------
# == Fixation job related paths ==
# --------------------------------

def get_fixation_config_json_path(config_folder: Path) -> Path:
    """
    Returns the default save/load path for the fixation config JSON file.

    Args:
        project_root (Path): Root directory of the project.

    Returns:
        Path: Full path to 'saved_configs/fixation_config.json'
    """
    path = Path(config_folder) / "fixation_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_fixation_temp_dir(processed_data_dir: Path) -> Path:
    """
    Returns the path to the temporary directory used to store intermediate fixation and saccade job results.
    Creates the directory if it doesn't exist.

    Args:
        processed_data_dir (Path): Base directory where processed data is stored.

    Returns:
        Path: Path to the 'temp' subdirectory.
    """
    temp_dir = Path(processed_data_dir) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

def get_job_file_path(project_root: Path, job_file_name: str) -> Path:
    """
    Returns the full path to the job array file inside the 'jobs/scripts' folder.
    Ensures the folder exists.

    Args:
        project_root (Path): Root of the project.
        job_file_name (str): Filename of the job array file (e.g., 'fixation_job_array.txt').

    Returns:
        Path: Full path to the job array file.
    """
    jobs_dir = Path(project_root) / "jobs" / "scripts"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir / job_file_name


def get_worker_python_script_path(project_root: Path, relative_script_path: str) -> Path:
    """
    Returns the full path to the Python worker script that is executed by the job array.
    """
    return Path(project_root) / relative_script_path



def get_fixation_job_result_path(temp_dir: Path, session: str, run: str, agent: str) -> Path:
    """
    Constructs the path to a temporary fixation result file for a given session/run/agent.

    Args:
        temp_dir (Path): Directory where temporary results are stored.
        session (str): Session name.
        run (str): Run number or ID.
        agent (str): Agent name (e.g., 'm1' or 'm2').

    Returns:
        Path: Path to the fixation .pkl file.
    """
    return Path(temp_dir) / f"fixations_{session}_{run}_{agent}.pkl"


def get_saccade_job_result_path(temp_dir: Path, session: str, run: str, agent: str) -> Path:
    """
    Constructs the path to a temporary saccade result file for a given session/run/agent.

    Args:
        temp_dir (Path): Directory where temporary results are stored.
        session (str): Session name.
        run (str): Run number or ID.
        agent (str): Agent name (e.g., 'm1' or 'm2').

    Returns:
        Path: Path to the saccade .pkl file.
    """
    return Path(temp_dir) / f"saccades_{session}_{run}_{agent}.pkl"


def get_job_out_dir(project_root: Path) -> Path:
    """
    Returns the top-level jobs output directory.
    """
    out_dir = Path(project_root) / "jobs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_log_dir(job_out_dir: Path) -> Path:
    """
    Returns the path to the log directory inside the jobs folder.
    """
    log_dir = Path(job_out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_sbatch_script_path(job_out_dir: Path, job_name: str) -> Path:
    """
    Returns the full path to the shell script generated by dSQ for sbatch submission.
    """
    scripts_dir = Path(job_out_dir) / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    return scripts_dir / f"dsq-joblist_{job_name}.sh"


# --------------------------
# == Fixation probability ==
# --------------------------

def get_fixation_probability_path(config):
    return Path(config.processed_data_dir) / "fix_prob_df.pkl"


def get_fixation_probability_by_interactivity_path(config):
    return Path(config.processed_data_dir) / "fix_prob_by_interactivity_df.pkl"

def get_fixation_probability_by_interactivity_segment_path(config):
    return Path(config.processed_data_dir) / "fix_prob_by_interactivity_segment_df.pkl"


def get_fixation_probability_plot_dir(config) -> str:
    """
    Return a dated output directory for saving fixation probability plots.
    e.g., root_dir/outputs/plots/fixation_probabilities/2025-05-27
    """
    path = add_date_dir_to_path(
        Path(config.plots_dir) / "fixation_probabilities",
    )
    os.makedirs(path, exist_ok=True)
    return path
    

# ---------------------------------
# == PC fit and projection paths ==
# ---------------------------------

def get_pc_model_basedir(config):
    return Path(config.output_dir) / "pc_projection"

def get_pc_fit_model_path(base_dir: str, fit_name: str, region: str) -> Path:
    """
    Returns the path to the saved PCA fit model file for a given fit name and brain region.

    Args:
        base_dir (str): Base directory where PCA results are stored.
        fit_name (str): Name of the PCA fitting specification.
        region (str): Brain region name (e.g., 'ofc').

    Returns:
        Path: Path to the fit model .pkl file.
    """
    return Path(base_dir) / fit_name / f"fit_model_{region}.pkl"


def get_pc_fit_orders_path(base_dir: str, fit_name: str, region: str) -> Path:
    """
    Returns the path to the unit and category order file for a given PCA fit.

    Args:
        base_dir (str): Base directory where PCA results are stored.
        fit_name (str): Name of the PCA fitting specification.
        region (str): Brain region name.

    Returns:
        Path: Path to the .pkl file containing order metadata.
    """
    return Path(base_dir) / fit_name / f"orders_{region}.pkl"


def get_pc_projection_path(base_dir: str, fit_name: str, transform_name: str, region: str) -> Path:
    """
    Returns the path to the projection result file for a given fit/transform pair and region.

    Args:
        base_dir (str): Base directory where PCA results are stored.
        fit_name (str): Name of the PCA fitting specification.
        transform_name (str): Name of the PCA transform specification.
        region (str): Brain region name.

    Returns:
        Path: Path to the projection .pkl file.
    """
    return Path(base_dir) / f"{fit_name}__{transform_name}" / f"projection_{region}.pkl"


def get_pc_projection_meta_path(base_dir: str, fit_name: str, transform_name: str) -> Path:
    """
    Returns the path to the metadata file associated with a given PCA projection.

    Args:
        base_dir (str): Base directory where PCA results are stored.
        fit_name (str): Name of the PCA fitting specification.
        transform_name (str): Name of the PCA transform specification.

    Returns:
        Path: Path to the projection metadata .json file.
    """
    return Path(base_dir) / f"{fit_name}__{transform_name}" / "meta.pkl"


def get_pc_trajectory_plots_base_dir(config):
    return Path(config.plots_dir) / "pc_trajectories"


def get_pc_trajectory_plot_dir_for_fit_transform_combination(base_dir, fit_name, transform_name, dated=True):
    if dated:
        base_dir = add_date_dir_to_path(base_dir)
    dir_path = os.path.join(base_dir, fit_name, transform_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def add_date_dir_to_path(base_path: str) -> str:
    """
    Appends today's date (YYYY-MM-DD) as a subdirectory to the given path.

    Args:
        base_path (str): The base directory path.

    Returns:
        str: The updated path with today's date appended.
    """
    today = date.today().isoformat()
    return os.path.join(base_path, today)


def get_pc_trajectory_comparison_plots_base_dir(config):
    return Path(config.plots_dir) / "pc_trajectory_comparison"


def get_pc_trajectory_comparison_plot_dir(base_dir, fit_name, dated=True):
    """
    Get the directory path for saving trajectory comparison plots for a given fit.
    """
    dir_path = os.path.join(add_date_dir_to_path(base_dir), fit_name) if dated else os.path.join(base_dir, fit_name)
    return dir_path

# ------------------------
# == General path tools ==
# ------------------------

def join_folder_and_filename(folder: Union[str, Path], filename: str) -> Path:
    """
    Joins a folder path and filename to produce a full file path.

    Args:
        folder (Union[str, Path]): The folder path.
        filename (str): The filename.

    Returns:
        Path: The combined file path.
    """
    return Path(folder) / filename


########################
## PSTH related paths ##
########################

class PSTHPaths:
    """
    Centralized path generator for PSTH dataframes and plots.
    """

    def __init__(self, config):
        self.config = config

    # === Processed data outputs ===

    def get_psth_output_dir(self) -> Path:
        """
        Base directory for processed PSTH .pkl files.
        """
        return Path(self.config.processed_data_dir) / "psth"

    def get_psth_per_trial_path(self) -> Path:
        return self.get_psth_output_dir() / "psth_per_trial.pkl"

    def get_avg_face_obj_path(self) -> Path:
        return self.get_psth_output_dir() / "avg_face_obj.pkl"

    def get_avg_int_non_int_face_path(self) -> Path:
        return self.get_psth_output_dir() / "avg_int_non_int_face.pkl"

    # === Plots ===

    def get_plots_dir(self) -> Path:
        """
        Root PSTH plots directory under the global plots_dir.
        """
        return Path(self.config.plots_dir) / "psth"

    def get_psth_plot_date_dir(self) -> Path:
        """
        Date-specific subfolder for PSTH plots.
        Example: <plots_dir>/psth/<YYYY-MM-DD>/
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.get_plots_dir() / date_str


####################################
## Crosscorrelation related paths ##
####################################

class CrossCorrPaths:
    """
    Centralized path and filename generator for cross-correlation analysis.
    All file naming conventions and directory logic should be encapsulated here.
    """

    def __init__(self, config):
        self.config = config


    # === Job script and worker path ===

    def get_job_file_path(self, job_file_name) -> Path:
        return self.config.project_root / "jobs" / "scripts" / job_file_name


    def get_worker_script_path(self) -> Path:
        return self.config.project_root / "scripts" / "behav_analysis" / "04_inter_agent_crosscorr.py"


    # === Output base directories ===

    def get_output_dir(self) -> Path:
        return Path(self.config.output_dir) / "crosscorr"


    def get_shuffled_output_dir(self) -> Path:
        return Path(self.config.output_dir) / "crosscorr_shuffled"


    def get_shuffled_temp_dir(self) -> Path:
        return self.get_shuffled_output_dir() / "temp"


    # === Observed cross-correlation filenames and paths ===

    def get_comparison_name(self, a1: str, b1: str, a2: str, b2: str) -> str:
        return f"{a1}_{b1}__vs__{a2}_{b2}"


    def get_obs_crosscorr_filename(self, a1, b1, a2, b2, period_type: str) -> str:
        name = self.get_comparison_name(a1, b1, a2, b2)
        name += f"__{period_type}"
        return f"{name}.pkl"


    def get_obs_crosscorr_path(self, a1, b1, a2, b2, period_type: str) -> Path:
        return self.get_output_dir() / self.get_obs_crosscorr_filename(a1, b1, a2, b2, period_type)


    def get_obs_crosscorr_path_by_name(self, comparison_name: str) -> Path:
        return self.get_output_dir() / f"{comparison_name}.pkl"


    # === Shuffled cross-correlation file logic ===

    def get_shuffled_temp_filename(self, session, run, a1, b1, a2, b2, period_type: str) -> str:
        name = self.get_comparison_name(a1, b1, a2, b2)
        return f"{name}__{period_type}__{session}__run{run}.pkl"


    def get_shuffled_temp_path(self, session, run, a1, b1, a2, b2, period_type: str) -> Path:
        return self.get_shuffled_temp_dir() / self.get_shuffled_temp_filename(session, run, a1, b1, a2, b2, period_type)


    def get_shuffled_final_filename(self, a1, b1, a2, b2, period_type: str) -> str:
        name = self.get_comparison_name(a1, b1, a2, b2)
        name += f"__{period_type}"
        return f"{name}.pkl"


    def get_shuffled_final_path(self, a1, b1, a2, b2, period_type: str) -> Path:
        return self.get_shuffled_output_dir() / self.get_shuffled_final_filename(a1, b1, a2, b2, period_type)


    def parse_shuffled_temp_filename(self, path: Path) -> Optional[Tuple[str, str, str, str, str, str, str]]:
        """
        Parses a shuffled temp filename and returns its components:
        (a1, b1, a2, b2, period_type, session, run)

        Example:
            m1_face_fixation__vs__m2_face_fixation__full__20200101__run3.pkl
        Returns:
            ('m1', 'face_fixation', 'm2', 'face_fixation', 'full', '20200101', '3')
        """
        parts = path.stem.split("__")

        if len(parts) != 6 or parts[1] != "vs" or parts[3] not in {"full", "interactive", "non_interactive"}:
            logger.warning(f"Unexpected temp filename format: {path.name}")
            return None

        try:
            a1, b1 = parts[0].split("_", 1)
            a2, b2 = parts[2].split("_", 1)
            period_type = parts[3]
            session = parts[4]
            run = parts[5].replace("run", "")
        except Exception as e:
            logger.warning(f"Failed to parse temp file '{path.name}': {e}")
            return None

        return a1, b1, a2, b2, period_type, session, run


    def get_grouped_shuffled_temp_paths(self) -> Dict[Tuple[str, str, str, str, str], List[Path]]:
        """
        Scans the temp shuffled directory and groups files by (a1, b1, a2, b2, period_type).
        
        Returns:
            Dictionary mapping group_key -> list of .pkl Paths.
        """
        grouped_paths: Dict[Tuple[str, str, str, str, str], List[Path]] = defaultdict(list)

        for path in self.get_shuffled_temp_dir().glob("*.pkl"):
            parsed = self.parse_shuffled_temp_filename(path)

            if parsed is None:
                continue
            a1, b1, a2, b2, period_type, session, run = parsed
            group_key = (a1, b1, a2, b2, period_type)
            grouped_paths[group_key].append(path)

        return grouped_paths


    # === Final results file ===

    def get_analysis_output_path(self) -> Path:
        return Path(self.config.output_dir) / "results" / "mean_minus_shuffled_crosscorr_results.pkl"


    def get_crosscorr_deltas_plot_dir(self) -> Path:
        """
        Returns the directory for Δ crosscorr plots.
        Structure: <output_dir>/plots/mean_minus_shuffled_crosscorr/<YYYY-MM-DD>/
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        return Path(self.config.output_dir) / "plots" / "mean_minus_shuffled_crosscorr" / date_str

