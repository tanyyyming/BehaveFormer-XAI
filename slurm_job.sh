#!/bin/bash
#SBATCH --gpus=1 --constraint=xgpf
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 15:00:00
#SBATCH -J behaveformer-humidb-train
#SBATCH -o slurm_logs/%x-%j.out

# initialize conda for this shell
source ~/anaconda3/etc/profile.d/conda.sh

# now you can activate envs
conda activate BehaveFormer

# ==== optional sanity checks ====
echo "Python: $(which python)"
# Slurm sets CUDA_VISIBLE_DEVICES for you; no need to override.
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ==== run your script ====s
bash scripts/humi/main/train.sh down all
