#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
##SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=8     # 2 CPU cores
#SBATCH --mem=128G             # GB memory
#SBATCH --output=three_months/all_vessels_2025/output_concat_2025_%j.txt    # Log file
#SBATCH --job-name=all_vessels_concat
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
python -u features_all_2025.py