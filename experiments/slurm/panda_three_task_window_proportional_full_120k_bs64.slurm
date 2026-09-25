#!/bin/bash
#SBATCH --job-name=molmoact2-panda-three-window-full-120k-b64
#SBATCH --account=ka
#SBATCH --partition=gpu_h100_il,gpu_a100_il
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

export RUN_NAME_OVERRIDE="panda-three-task-window-proportional-full-bs64-4gpu-120k"
export RUN_GROUP_OVERRIDE="panda-three-task-weighting"
export TRAINING_MIXTURE_OVERRIDE="panda_three_task_window_proportional"
export CHECKPOINT_NAME_OVERRIDE="panda-three-task-window-proportional-full-h30s30-bs64-4gpu-120k"
export SLURM_SCRIPT_OVERRIDE="$HOME/molmoact2/experiments/slurm/panda_three_task_window_proportional_full_120k_bs64.slurm"

exec bash "$HOME/molmoact2/experiments/slurm/panda_three_task_absolute_full_120k_bs64.slurm"
