#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
##SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=8     # 2 CPU cores
#SBATCH --mem=256G             # GB memory
#SBATCH --output=training_FINAL/test_vessels_2024/output_metric_2024_%j.txt    # Log file
#SBATCH --job-name=metrics_seen_2024
#SBATCH --mail-user=trulsc00@gmail.com
#SBATCH --mail-type=ALL

# Go to submission folder
cd ~/MODELS/LSTM

# Load Python
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Activate venv
source ~/venv/py311/bin/activate


# Run your script
python -u metrics_2024_seen_full.py