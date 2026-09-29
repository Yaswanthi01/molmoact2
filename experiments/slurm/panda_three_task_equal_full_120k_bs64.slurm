#!/bin/bash
#SBATCH --job-name=molmoact2-panda-three-equal-full-120k-b64
#SBATCH --account=ka
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

export RUN_NAME_OVERRIDE="panda-three-task-equal-full-bs64-4gpu-120k"
export RUN_GROUP_OVERRIDE="panda-three-task-weighting"
export TRAINING_MIXTURE_OVERRIDE="panda_three_task_equal"
export CHECKPOINT_NAME_OVERRIDE="panda-three-task-equal-full-h30s30-bs64-4gpu-120k"
export SLURM_SCRIPT_OVERRIDE="$HOME/molmoact2/experiments/slurm/panda_three_task_equal_full_120k_bs64.slurm"
export DEVICE_BATCH_SIZE_OVERRIDE="${DEVICE_BATCH_SIZE_OVERRIDE:-8}"
export TRAINING_TIME_LIMIT_OVERRIDE="${TRAINING_TIME_LIMIT_OVERRIDE:-1200}"

exec bash "$HOME/molmoact2/experiments/slurm/panda_three_task_absolute_full_120k_bs64.slurm"
