#!/bin/bash
#SBATCH --job-name=molmoact2-sort3-full-1g-tonight
#SBATCH --account=ka
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

export RUN_NAME_OVERRIDE="panda-sort-three-absolute-all-full-bs64-1gpu-tonight"
export NPROC_PER_NODE=1
export TRAIN_TIME_LIMIT=1200
export BATCH_LAYOUT="1 GPU, global batch 64, device microbatch 4, gradient accumulation 16"
export SLURM_SCRIPT_OVERRIDE="$HOME/molmoact2/experiments/slurm/panda_sort_three_absolute_all_full_120k_bs64_1gpu_tonight.slurm"

exec bash "$HOME/molmoact2/experiments/slurm/panda_sort_three_absolute_all_full_120k_bs64.slurm"
