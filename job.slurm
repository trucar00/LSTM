#!/bin/bash
#SBATCH --account=share-ie-itk
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0:5:00
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=16     # 2 CPU cores
#SBATCH --mem=64G             # GB memory
#SBATCH --output=output.txt    # Log file
#SBATCH --mail-user=trulsc00@gmail.com
#SBATCH --mail-type=ALL

# Go to submission folder
cd ~/MODELS/LSTM

# Load Python
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Activate venv
source ~/venv/myenv/bin/activate

echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi

# Run your script
python model2.py