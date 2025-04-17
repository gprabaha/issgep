#!/bin/bash
#SBATCH --job-name=fixation_detection
#SBATCH --partition=psych_day
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=jobs/logs/00_fixation_main_job.out
#SBATCH --error=jobs/logs/00_fixation_main_job.err

# Load environment
module load miniconda
conda deactivate
conda activate gaze_processing

# Run your script
python scripts/analysis/01_fixation_detection.py
