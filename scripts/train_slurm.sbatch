#!/usr/bin/env bash
# Slurm wrapper around scripts/train.sh. Fill in the partition/account for your cluster, then:
#
#   sbatch scripts/train_slurm.sbatch configs/generated/rlvr_only/recap.yaml
#
#SBATCH --job-name=recap
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=72:00:00
### Cluster-specific -- uncomment and set these for your site:
### SBATCH --partition=<your-partition>
### SBATCH --account=<your-account>
set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/train_slurm.sbatch <config.yaml>}"

mkdir -p logs
nvidia-smi

# Activate your environment here, e.g.:
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate recap

# HF_TOKEN / WANDB_API_KEY should come from your shell profile or a secrets manager,
# not from this file.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

bash scripts/train.sh "$CONFIG"
