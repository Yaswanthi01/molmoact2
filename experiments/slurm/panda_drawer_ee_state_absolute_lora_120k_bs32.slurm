#!/bin/bash
#SBATCH --job-name=molmoact2-panda-drawer-abs-lora-120k-b32
#SBATCH --account=ka
#SBATCH --partition=gpu_h100_short,gpu_a100_il,gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:28:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

RUN_NAME="panda-drawer-absolute-lora-bs32-120k"
RUN_GROUP="panda-drawer-absolute"
CHAIN_ROOT_JOB_ID="${CHAIN_ROOT_JOB_ID:-$SLURM_JOB_ID}"
CHAIN_LOG_DIR="$HOME/molmoact2/logs/training/$RUN_GROUP/$RUN_NAME/chain-$CHAIN_ROOT_JOB_ID"
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

MOLMO_WORKSPACE="${MOLMO_WORKSPACE:?Set MOLMO_WORKSPACE to your allocated workspace path before sbatch}"
if [[ "$MOLMO_WORKSPACE" != /* || ! -d "$MOLMO_WORKSPACE" || ! -w "$MOLMO_WORKSPACE" ]]; then
  echo "MOLMO_WORKSPACE must be an absolute, existing, writable directory: $MOLMO_WORKSPACE" >&2
  exit 1
fi

DATASET_ROOT="$HOME/molmoact2/datasets/panda_drawer_ee_state_absolute_molmoact2_v3"
CHECKPOINT_ROOT="$MOLMO_WORKSPACE/checkpoints"
SAVE_FOLDER="$CHECKPOINT_ROOT/panda-drawer-ee-state-absolute-lora-h30s30-bs32-120k"
MILESTONE_FOLDER="$CHECKPOINT_ROOT/panda-drawer-ee-state-absolute-lora-h30s30-bs32-120k-milestones"
TARGET_STEPS=120000
SLURM_SCRIPT="$HOME/molmoact2/experiments/slurm/panda_drawer_ee_state_absolute_lora_120k_bs32.slurm"

mkdir -p logs "$SAVE_FOLDER" "$MILESTONE_FOLDER" "$HF_HOME"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Required dataset not found: $DATASET_ROOT" >&2
  echo "Create it first with:" >&2
  echo "  python scripts/prepare_panda_drawer_ee_state_dataset.py --variant absolute --symlink-videos" >&2
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

echo "Training mixture: panda_drawer_ee_state_absolute"
echo "Training mode: LoRA VLM with fully trainable action expert"
echo "Action representation: absolute end-effector pose"
echo "Training target: step${TARGET_STEPS}"
echo "Batch layout: 1 GPU, global batch 32, device microbatch 8, gradient accumulation 4"
if (( starting_step == 0 )); then
  echo "Starting fresh from base checkpoint: allenai/MolmoAct2"
else
  echo "Resuming from: step${starting_step}"
fi
echo "Checkpoint folder: ${SAVE_FOLDER}"

torchrun --standalone --nproc-per-node=1 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  panda_drawer_ee_state_absolute \
  --frame_loading_backend=av \
  --max_duration=120000 \
  --scheduler.t_max=120000 \
  --time_limit=1200 \
  --canceled_check_interval=1 \
  --log_interval=10 \
  --device_batch_size=8 \
  --global_batch_size=32 \
  --num_workers=4 \
  --pin_memory=true \
  --run_name="$RUN_NAME" \
  --wandb.group="$RUN_GROUP" \
  --save_folder="$SAVE_FOLDER" \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=2 \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=true \
  --ft_action_expert=true \
  --ft_embedding=lm_head \
  --lora_enable=true \
  --lora_rank=64 \
  --lora_alpha=16 \
  --lora_dropout=0.05 \
  --connector_learning_rate=5e-5 \
  --vit_learning_rate=5e-5 \
  --llm_learning_rate=5e-5 \
  --action_expert_learning_rate=5e-5

for milestone in 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000; do
  for suffix in "" "-lora-llm" "-lora-vision" "-merged"; do
    source_checkpoint="$SAVE_FOLDER/step${milestone}${suffix}"
    archived_checkpoint="$MILESTONE_FOLDER/step${milestone}${suffix}"
    if [[ -d "$source_checkpoint" && ! -e "$archived_checkpoint" ]]; then
      echo "Preserving evaluation milestone step${milestone}${suffix}"
      cp -al "$source_checkpoint" "$archived_checkpoint"
    fi
  done
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
