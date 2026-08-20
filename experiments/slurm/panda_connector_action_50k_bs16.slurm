#!/bin/bash
#SBATCH --job-name=molmoact2-panda-conn-action-50k-b16
#SBATCH --account=ka
#SBATCH --partition=gpu_h100_short,gpu_a100_il,gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:28:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/molmoact2/experiments"
source "$HOME/molmoact2/.venv_molmoact312/bin/activate"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Python: $(which python)"
python --version
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

export LEROBOT_DATA_ROOT="$HOME/molmoact2/datasets"
export MOLMO_DATA_DIR="$HOME/molmoact2/datasets"
export LEROBOT_VIDEO_BACKEND=pyav
export HF_HOME="$HOME/molmoact2/.cache/huggingface"
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}"
export WANDB_PROJECT="molmoact2-panda-connector-action-50k-b16"
export WANDB_ENTITY="local"
export WANDB_MODE="offline"

SAVE_FOLDER="$HOME/molmoact2/checkpoints/panda-ee-connector-action-h30s30-bs16-scratch-50k"
MILESTONE_FOLDER="$HOME/molmoact2/checkpoints/panda-ee-connector-action-h30s30-bs16-scratch-50k-milestones"
TARGET_STEPS=50000
SLURM_SCRIPT="$HOME/molmoact2/experiments/slurm/panda_connector_action_50k_bs16.slurm"

mkdir -p logs "$SAVE_FOLDER" "$MILESTONE_FOLDER" "$HF_HOME"

starting_step=0
for checkpoint_dir in "$SAVE_FOLDER"/step*; do
  [[ -d "$checkpoint_dir" ]] || continue
  step_name="${checkpoint_dir##*/step}"
  if [[ "$step_name" =~ ^[0-9]+$ ]] && (( step_name > starting_step )); then
    starting_step="$step_name"
  fi
done

echo "Training target: step${TARGET_STEPS}"
if (( starting_step == 0 )); then
  echo "Starting fresh from base checkpoint: allenai/MolmoAct2"
else
  echo "Resuming from: step${starting_step}"
fi
echo "Checkpoint folder: ${SAVE_FOLDER}"

torchrun --standalone --nproc-per-node=1 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  panda_drawer_ee \
  --frame_loading_backend=av \
  --max_duration=50000 \
  --scheduler.t_max=50000 \
  --time_limit=1200 \
  --canceled_check_interval=1 \
  --log_interval=10 \
  --device_batch_size=1 \
  --global_batch_size=16 \
  --num_workers=0 \
  --pin_memory=false \
  --save_folder="$SAVE_FOLDER" \
  --save_interval=5000 \
  --save_num_checkpoints_to_keep=3 \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none \
  --connector_learning_rate=5e-6 \
  --action_expert_learning_rate=5e-5 \
  --ft_connector=true

for milestone in 35000 50000; do
  source_checkpoint="$SAVE_FOLDER/step${milestone}"
  archived_checkpoint="$MILESTONE_FOLDER/step${milestone}"
  if [[ -d "$source_checkpoint" && ! -e "$archived_checkpoint" ]]; then
    echo "Preserving evaluation milestone step${milestone}"
    cp -al "$source_checkpoint" "$archived_checkpoint"
  fi
done

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
  echo "Target is step${TARGET_STEPS}; submitting the next segment."
  next_job_output="$(sbatch --dependency="afterok:${SLURM_JOB_ID}" "$SLURM_SCRIPT")"
  echo "$next_job_output"
else
  echo "Training target reached at step${latest_step}."
fi
