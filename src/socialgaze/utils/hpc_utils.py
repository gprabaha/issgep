# src/socialgaze/utils/hpc_utils.py

import os
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

def generate_fixation_job_file(tasks: List[Tuple[str, str, str]], job_file_path: Path, script_path: str, is_grace: bool):
    """
    Generate a job file that runs fixation detection with one line per session/run/agent.
    """
    os.makedirs(job_file_path.parent, exist_ok=True)
    env_name = "gaze_otnal" if is_grace else "gaze_processing"

    with open(job_file_path, "w") as f:
        for session, run, agent in tasks:
            cmd = (
                f"module load miniconda; "
                f"conda activate {env_name}; "
                f"python {script_path} --session {session} --run {run} --agent {agent}"
            )
            f.write(cmd + "\n")
    logger.info("Wrote fixation job file to %s", job_file_path)


def submit_dsq_array_job(job_file_path: Path, job_out_dir: Path, job_name: str = "fixation", partition: str = "psych_day", cpus: int = 8, mem_per_cpu: int = 1000, time_limit: str = "01:00:00"):
    """
    Submit a DSQ job array and return the job ID.
    """
    job_script_path = job_out_dir / "scripts" / f'dsq-joblist_{job_name}.sh'
    log_dir = job_out_dir / "logs"
    os.makedirs(job_out_dir, exist_ok=True)

    logger.info("Generating dSQ job script for job file %s", job_file_path)
    subprocess.run(
        f'module load dSQ; dsq --job-file {job_file_path} --batch-file {job_script_path} '
        f'-o {log_dir} --status-dir {log_dir} --partition {partition} '
        f'--cpus-per-task {cpus} --mem-per-cpu {mem_per_cpu} -t {time_limit} --mail-type FAIL',
        shell=True, check=True, executable='/bin/bash'
    )

    if not job_script_path.exists():
        raise RuntimeError(f"dSQ job script was not created: {job_script_path}")

    logger.info("Submitting job array via sbatch")
    result = subprocess.run(
        f'sbatch --job-name=dsq_{job_name} '
        f'--output={log_dir}/fixation_%a.out '
        f'--error={log_dir}/fixation_%a.err '
        f'{job_script_path}',
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
