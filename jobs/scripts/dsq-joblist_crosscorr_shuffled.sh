#!/bin/bash
#SBATCH --output /gpfs/milgram/pi/chang/pg496/repositories/issgep/jobs/logs
#SBATCH --array 0-4169
#SBATCH --job-name dsq-crosscorr_shuffled_job_array
#SBATCH --partition psych_day --cpus-per-task 8 --mem-per-cpu 4000 -t 00:05:00 --mail-type FAIL

# DO NOT EDIT LINE BELOW
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/milgram/pi/chang/pg496/repositories/issgep/jobs/scripts/crosscorr_shuffled_job_array.txt --status-dir /gpfs/milgram/pi/chang/pg496/repositories/issgep/jobs/logs

