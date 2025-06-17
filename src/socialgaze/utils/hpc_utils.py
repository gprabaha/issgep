# src/socialgaze/utils/hpc_utils.py

import os
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def generate_fixation_job_file(tasks: List[Tuple[str, str, str]], config) -> None:
    """
    Generate a job array file with one command per (session, run, agent) for fixation detection.
    Uses environment and script paths from the FixationConfig.

    Args:
        tasks (List[Tuple[str, str, str]]): List of (session, run, agent) tuples.
        config (FixationConfig): Config object containing paths and environment info.
    """
    job_file_path = config.job_file_path
    worker_script = config.worker_python_script_path
    env_name = config.env_name

    os.makedirs(job_file_path.parent, exist_ok=True)

    with open(job_file_path, "w") as f:
        for session, run, agent in tasks:
            cmd = (
                f"module load miniconda; "
                f"conda activate {env_name}; "
                f"python {worker_script} --session {session} --run {run} --agent {agent}"
            )
            f.write(cmd + "\n")

    logger.info("Wrote fixation job file to %s", job_file_path)


def generate_crosscorr_job_file(tasks: List[Tuple[str, int, str, str, str, str, str]], config) -> None:
    """
    Generate a job array file with one command per (session, run, a1, b1, a2, b2, period_type).

    Args:
        tasks: List of (session, run, a1, b1, a2, b2, period_type) tuples.
        config: CrossCorrConfig object with env_name, job_file_path, etc.
    """
    job_file_path = config.job_file_path
    worker_script = config.worker_python_script_path
    env_name = config.env_name

    os.makedirs(job_file_path.parent, exist_ok=True)

    with open(job_file_path, "w") as f:
        for session, run, a1, b1, a2, b2, period_type in tasks:
            cmd = (
                f"module load miniconda; "
                f"conda activate {env_name}; "
                f"python {worker_script} "
                f"--session {session} --run {run} "
                f"--a1 {a1} --b1 {b1} --a2 {a2} --b2 {b2} "
                f"--mode shuffled --period_type {period_type}"
            )
            f.write(cmd + "\n")

    logger.info("Wrote shuffled cross-correlation job file to %s", job_file_path)




def submit_dsq_array_job(config) -> str:
    """
    Submits a dSQ job array using parameters from the FixationConfig.

    Args:
        config (FixationConfig): Config object containing job, script, and log paths.

    Returns:
        str: SLURM job ID of the submitted job array.
    """
    logger.info("Generating dSQ job script for job file %s", config.job_file_path)

    subprocess.run(
        f'module load dSQ; dsq --job-file {config.job_file_path} --batch-file {config.sbatch_script_path} '
        f'-o {config.log_dir} --status-dir {config.log_dir} --partition {config.partition} '
        f'--cpus-per-task {config.cpus_per_task} --mem-per-cpu {config.mem_per_cpu} '
        f'-t {config.time_limit} --mail-type FAIL',
        shell=True, check=True, executable='/bin/bash'
    )

    if not config.sbatch_script_path.exists():
        raise RuntimeError(f"dSQ job script was not created: {config.sbatch_script_path}")

    logger.info("Submitting job array via sbatch")
    result = subprocess.run(
        f'sbatch --job-name=dsq_{config.job_name} '
        f'--output={config.log_dir}/fixation_%a.out '
        f'--error={config.log_dir}/fixation_%a.err '
        f'{config.sbatch_script_path}',
        shell=True, check=True, capture_output=True, text=True, executable='/bin/bash'
    )

    job_id = result.stdout.strip().split()[-1]
    logger.info(f"Submitted job array with ID: {job_id}")
    return job_id



def track_job_completion(job_id: str, poll_secs: int = 30, log_every_secs: int = 60):
    """
    Track job array status using `squeue`.
    """
    logger.info("Tracking job array with ID: %s", job_id)
    start = time.time()
    last_log = start

    while True:
        result = subprocess.run(
            f'squeue --job {job_id} -h -o %T',
            shell=True, capture_output=True, text=True, executable='/bin/bash'
        )

        if result.returncode != 0:
            logger.error("Failed to check job status: %s", result.stderr.strip())
            break

        statuses = result.stdout.strip().split()
        if not statuses or all(s not in {"PENDING", "RUNNING", "CONFIGURING"} for s in statuses):
            logger.info("Job array %s has completed.", job_id)
            break

        if time.time() - last_log >= log_every_secs:
            logger.info("Still running... job array %s", job_id)
            last_log = time.time()

        time.sleep(poll_secs)
