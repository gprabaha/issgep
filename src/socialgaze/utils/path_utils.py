# src/socialgaze/utils/path_utils.py

from pathlib import Path
from typing import Dict, Optional, List, Union

# --------------
# Config
# --------------

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


def get_spike_df_pkl_path(config) -> Path:
    """
    Returns the path to the saved spike dataframe as .pkl.
    """
    return config.processed_data_dir / "spike_data.pkl"


# ----------------------
# Fixation jobs
# ----------------------

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



def get_mutual_fixation_density_path(config, fixation_type='face') -> Path:
    return config.processed_data_dir / f"mutual_fixation_density_{fixation_type}.pkl"

def get_interactivity_df_path(config, fixation_type='face') -> Path:
    return config.processed_data_dir / "interactive_periods.pkl"


# --------------
# General
# --------------

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
