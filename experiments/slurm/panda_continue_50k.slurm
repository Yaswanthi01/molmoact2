#!/bin/bash
#SBATCH --job-name=molmoact2-panda-ee-50k
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
export WANDB_PROJECT="molmoact2-panda-ee-50k"
export WANDB_ENTITY="local"
export WANDB_MODE="offline"

# Seed a separate continuation folder from the completed baseline. This keeps
# checkpoint rotation in the 50k run from deleting or changing step8878.
BASELINE_FOLDER="$HOME/molmoact2/checkpoints/panda-ee-full-1epoch"
SAVE_FOLDER="$HOME/molmoact2/checkpoints/panda-ee-50k"
MILESTONE_FOLDER="$HOME/molmoact2/checkpoints/panda-ee-50k-milestones"
TARGET_STEPS=50000
SLURM_SCRIPT="$HOME/molmoact2/experiments/slurm/panda_continue_50k.slurm"

mkdir -p logs "$SAVE_FOLDER" "$MILESTONE_FOLDER" "$HF_HOME"

# On the first submission only, seed the new run with hard links to the
# completed baseline checkpoint. Checkpoint files are immutable, so this avoids
# duplicating their storage while keeping the original path intact.
if [[ ! -d "$SAVE_FOLDER/step8878" ]]; then
  if [[ ! -d "$BASELINE_FOLDER/step8878" ]]; then
    echo "Required baseline checkpoint not found: $BASELINE_FOLDER/step8878" >&2
    exit 1
  fi
  echo "Seeding 50k continuation from $BASELINE_FOLDER/step8878"
  cp -al "$BASELINE_FOLDER/step8878" "$SAVE_FOLDER/step8878"
  if [[ -f "$BASELINE_FOLDER/config.yaml" && ! -e "$SAVE_FOLDER/config.yaml" ]]; then
    cp -a "$BASELINE_FOLDER/config.yaml" "$SAVE_FOLDER/config.yaml"
  fi
fi

starting_step=0
for checkpoint_dir in "$SAVE_FOLDER"/step*; do
  [[ -d "$checkpoint_dir" ]] || continue
  step_name="${checkpoint_dir##*/step}"
  if [[ "$step_name" =~ ^[0-9]+$ ]] && (( step_name > starting_step )); then
    starting_step="$step_name"
  fi
done

if (( starting_step < 8878 )); then
  echo "No completed checkpoint at or beyond step8878 was found in $SAVE_FOLDER" >&2
  exit 1
fi

echo "Training target: step${TARGET_STEPS}"
echo "Starting/resuming from: step${starting_step}"
echo "Resume/checkpoint folder: ${SAVE_FOLDER}"

torchrun --standalone --nproc-per-node=1 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  panda_drawer_ee \
  --frame_loading_backend=av \
  --max_duration=50000 \
  --time_limit=1500 \
  --canceled_check_interval=1 \
  --log_interval=10 \
  --device_batch_size=1 \
  --global_batch_size=8 \
  --num_workers=0 \
  --pin_memory=false \
  --save_folder="$SAVE_FOLDER" \
  --save_interval=5000 \
  --save_num_checkpoints_to_keep=3 \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none

# Preserve exact evaluation milestones outside the rotating resume-checkpoint
# set. Hard links avoid duplicating the large checkpoint files while ensuring
# that cleanup of the original step directory does not remove the milestone.
for milestone in 10000 15000 20000 25000 30000 35000 40000 45000 50000; do
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
