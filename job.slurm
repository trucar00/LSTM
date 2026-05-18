#!/bin/bash
#SBATCH --account=share-ie-itk
#SBATCH --partition=CPUQ
# #SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=32     # 2 CPU cores
#SBATCH --mem=128G             # GB memory
#SBATCH --output=output_features_biggy.txt    # Log file
#SBATCH --mail-user=trulsc00@gmail.com
#SBATCH --mail-type=ALL

# Go to submission folder
cd ~/MODELS/LSTM

# Load Python
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Activate venv
source ~/venv/myenv/bin/activate # VENV FOR GPU

echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python - <<'EOF'
import torch, sys
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
EOF

nvidia-smi

# Run your script
python features_2_conf_big.py