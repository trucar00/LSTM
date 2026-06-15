#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
##SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=8     # 2 CPU cores
#SBATCH --mem=32G             # GB memory
#SBATCH --output=COUNT_GEAR_%j.txt    # Log file
#SBATCH --mail-user=trulsc00@gmail.com
#SBATCH --mail-type=ALL

# Go to submission folder
cd ~/MODELS/LSTM

# Load Python
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Activate venv
source ~/venv/py311/bin/activate # VENV FOR GPU


# Run your script
python -u count_gear_mmsis.py