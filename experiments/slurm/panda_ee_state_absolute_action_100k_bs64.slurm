#!/bin/bash
#SBATCH --job-name=molmoact2-panda-ee-state-abs-100k-b64
#SBATCH --account=ka
#SBATCH --partition=gpu_h100_short,gpu_a100_il,gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:28:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

CHAIN_ROOT_JOB_ID="${CHAIN_ROOT_JOB_ID:-$SLURM_JOB_ID}"
CHAIN_LOG_DIR="$HOME/molmoact2/logs/$CHAIN_ROOT_JOB_ID"
mkdir -p "$CHAIN_LOG_DIR"
exec > >(tee -a "$CHAIN_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out") \
  2> >(tee -a "$CHAIN_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2)

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
export WANDB_PROJECT="molmoact2"
export WANDB_ENTITY="yaswanthi2001-karlsruhe-institute-of-technology"
export WANDB_MODE="online"

DATASET_ROOT="$HOME/molmoact2/datasets/panda_drawer_ee_state_absolute_molmoact2_v3"
SOURCE_RUN_FOLDER="$HOME/molmoact2/checkpoints/panda-ee-state-absolute-action-h30s30-bs32-50k"
SOURCE_MILESTONE_FOLDER="$HOME/molmoact2/checkpoints/panda-ee-state-absolute-action-h30s30-bs32-50k-milestones"
SAVE_FOLDER="$HOME/molmoact2/checkpoints/panda-ee-state-absolute-action-h30s30-bs64-from-bs32-step60000-100k"
MILESTONE_FOLDER="$HOME/molmoact2/checkpoints/panda-ee-state-absolute-action-h30s30-bs64-from-bs32-step60000-100k-milestones"
TARGET_STEPS=100000
SLURM_SCRIPT="$HOME/molmoact2/experiments/slurm/panda_ee_state_absolute_action_100k_bs64.slurm"

mkdir -p logs "$SAVE_FOLDER" "$MILESTONE_FOLDER" "$HF_HOME"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Required dataset not found: $DATASET_ROOT" >&2
  echo "Create it first with:" >&2
  echo "  python experiments/scripts/prepare_panda_drawer_ee_state_dataset.py --variant absolute --symlink-videos" >&2
  exit 1
fi

SOURCE_CHECKPOINT="$SOURCE_RUN_FOLDER/step60000"
if [[ ! -d "$SOURCE_CHECKPOINT" ]]; then
  SOURCE_CHECKPOINT="$SOURCE_MILESTONE_FOLDER/step60000"
fi
if [[ ! -d "$SOURCE_CHECKPOINT" ]]; then
  echo "Required batch-32 source checkpoint not found." >&2
  echo "Checked:" >&2
  echo "  $SOURCE_RUN_FOLDER/step60000" >&2
  echo "  $SOURCE_MILESTONE_FOLDER/step60000" >&2
  exit 1
fi

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
  echo "Branching from batch-32 checkpoint: $SOURCE_CHECKPOINT"
else
  echo "Resuming batch-64 branch from: step${starting_step}"
fi
echo "Checkpoint folder: ${SAVE_FOLDER}"

if (( starting_step >= TARGET_STEPS )); then
  echo "Training target already reached."
  exit 0
fi

# Restore the complete optimizer and trainer state so the first batch-64 update is
# step60001. The destination folder takes precedence on later automatic resumes.
torchrun --standalone --nproc-per-node=1 \
  launch_scripts/train_lerobot.py \
  "$SOURCE_CHECKPOINT" \
  panda_drawer_ee_state_absolute \
  --frame_loading_backend=av \
  --max_duration=100000 \
  --scheduler.t_max=60000 \
  --time_limit=1200 \
  --canceled_check_interval=1 \
  --log_interval=10 \
  --device_batch_size=8 \
  --global_batch_size=64 \
  --num_workers=4 \
  --pin_memory=true \
  --run_name=panda-drawer-absolute-bs64-from-bs32-step60k-to100k \
  --model.action_expert.mlp_ratio=4.0 \
  --save_folder="$SAVE_FOLDER" \
  --load_path="$SOURCE_CHECKPOINT" \
  --reset_optimizer_state=false \
  --reset_trainer_state=false \
  --save_interval=20000 \
  --save_num_checkpoints_to_keep=2 \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none

for milestone in 80000 100000; do
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
  next_job_output="$(sbatch \
    --dependency="afterok:${SLURM_JOB_ID}" \
    --export="ALL,CHAIN_ROOT_JOB_ID=${CHAIN_ROOT_JOB_ID}" \
    "$SLURM_SCRIPT")"
  echo "$next_job_output"
else
  echo "Training target reached at step${latest_step}."
fi
