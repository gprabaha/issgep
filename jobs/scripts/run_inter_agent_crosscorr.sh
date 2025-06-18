#!/bin/bash
#SBATCH --job-name=inter_agent_crosscorr
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6G
#SBATCH --time=5:00:00
#SBATCH --partition=psych_day
#SBATCH --output=jobs/logs/main_shuffled_crosscorr_job.out
#SBATCH --error=jobs/logs/main_shuffled_crosscorr_job.err

# Load modules and activate conda
module load miniconda
conda deactivate
conda activate gaze_processing

# Move to repo root
cd /gpfs/milgram/project/chang/pg496/repositories/issgep

# Set PYTHONPATH to src/ so socialgaze is discoverable
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Run the script
python scripts/behav_analysis/04_inter_agent_crosscorr.py
