# src/socialgaze/utils/parallel_utils.py

from joblib import Parallel
from tqdm import tqdm


def run_joblib_parallel(jobs, n_jobs=-1, verbose=10):
    """
    Run joblib-delayed jobs with tqdm-based progress bar.
    Args:
        jobs: list of delayed() job objects
        n_jobs: number of parallel jobs (e.g., from config.num_cpus)
        verbose: verbosity level for tqdm
    """
    jobs = list(jobs)
    return Parallel(n_jobs=n_jobs)(
        tqdm(jobs, total=len(jobs), desc="Running jobs", dynamic_ncols=True)
    )
