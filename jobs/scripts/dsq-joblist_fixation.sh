#!/bin/bash
#SBATCH --output /gpfs/milgram/pi/chang/pg496/repositories/issgep/jobs/logs
#SBATCH --array 0-833
#SBATCH --job-name dsq-fixation_job_array
#SBATCH --partition psych_day --cpus-per-task 1 --mem-per-cpu 8000 -t 00:15:00 --mail-type FAIL

# DO NOT EDIT LINE BELOW
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/milgram/pi/chang/pg496/repositories/issgep/jobs/scripts/fixation_job_array.txt --status-dir /gpfs/milgram/pi/chang/pg496/repositories/issgep/jobs/logs

