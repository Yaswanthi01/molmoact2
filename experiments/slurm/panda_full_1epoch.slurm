#!/bin/bash
#SBATCH --job-name=molmoact2-panda-ee-full-1ep
#SBATCH --account=ka
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/molmoact2/experiments"
source "$HOME/molmoact2/.venv_molmoact312/bin/activate"

echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

export LEROBOT_DATA_ROOT="$HOME/molmoact2/datasets"
export MOLMO_DATA_DIR="$HOME/molmoact2/datasets"
export LEROBOT_VIDEO_BACKEND=pyav
export HF_HOME="$HOME/molmoact2/.cache/huggingface"
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}"
export WANDB_PROJECT="molmoact2-panda-ee-full-1epoch"
export WANDB_ENTITY="local"
export WANDB_MODE="offline"

SAVE_FOLDER="$HOME/molmoact2/checkpoints/panda-ee-full-1epoch"
TARGET_STEPS=8878
SLURM_SCRIPT="$HOME/molmoact2/experiments/slurm/panda_full_1epoch.slurm"
mkdir -p logs "$SAVE_FOLDER" "$HF_HOME"

# The dataset has 71,019 frames. At global batch size 8, 8,878 optimizer
# steps cover approximately one epoch. The trainer stops after 1,400 seconds
# and saves a resumable checkpoint, leaving time before SLURM's 30-minute kill.
torchrun --standalone --nproc-per-node=1 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  panda_drawer_ee \
  --frame_loading_backend=av \
  --max_duration=8878 \
  --time_limit=1400 \
  --canceled_check_interval=1 \
  --log_interval=10 \
  --device_batch_size=1 \
  --global_batch_size=8 \
  --num_workers=0 \
  --pin_memory=false \
  --save_folder="$SAVE_FOLDER" \
  --save_interval=500 \
  --save_num_checkpoints_to_keep=3 \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none

latest_step=0
for checkpoint_dir in "$SAVE_FOLDER"/step*; do
  [[ -d "$checkpoint_dir" ]] || continue
  step_name="${checkpoint_dir##*/step}"
  if [[ "$step_name" =~ ^[0-9]+$ ]] && (( step_name > latest_step )); then
    latest_step="$step_name"
  fi
done

echo "Latest completed checkpoint: step${latest_step}"

if (( latest_step < TARGET_STEPS )); then
  echo "Training target is step${TARGET_STEPS}; submitting the next segment."
  next_job_output="$(sbatch --dependency="afterok:${SLURM_JOB_ID}" "$SLURM_SCRIPT")"
  echo "$next_job_output"
else
  echo "Full one-epoch fine-tuning target reached at step${latest_step}."
fi
