#!/bin/bash
#SBATCH --job-name=molmoact2-panda-ee-full-eval
#SBATCH --account=ka
#SBATCH --partition=gpu_h100_short,gpu_a100_il,gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:28:00
#SBATCH --array=0-100%10
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "$HOME/molmoact2/experiments"
source "$HOME/molmoact2/.venv_molmoact312/bin/activate"

echo "Host: $(hostname)"
echo "Job: ${SLURM_ARRAY_JOB_ID}"
echo "Episode: ${SLURM_ARRAY_TASK_ID}"
echo "Python: $(which python)"
python --version
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

export LEROBOT_DATA_ROOT="$HOME/molmoact2/datasets"
export MOLMO_DATA_DIR="$HOME/molmoact2/datasets"
export LEROBOT_VIDEO_BACKEND=pyav
export HF_HOME="$HOME/molmoact2/.cache/huggingface"
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}"

DATASET="panda_drawer_ee_molmoact2_v3"
DATASET_ROOT="$HOME/molmoact2/datasets/$DATASET"
CHECKPOINT="$HOME/molmoact2/checkpoints/panda-ee-full-1epoch/step8878"
EVALUATION_ROOT="$HOME/molmoact2/evaluations/panda-ee-full-step8878"
EPISODE_IDX="$SLURM_ARRAY_TASK_ID"
OUTPUT_DIR="$(printf '%s/episode%03d' "$EVALUATION_ROOT" "$EPISODE_IDX")"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Dataset not found: $DATASET_ROOT" >&2
  exit 1
fi

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Available completed checkpoints:" >&2
  find "$HOME/molmoact2/checkpoints/panda-ee-full-1epoch" \
    -maxdepth 1 -type d -name 'step[0-9]*' -print >&2 || true
  exit 1
fi

mkdir -p logs "$OUTPUT_DIR" "$HF_HOME"

# No --episode_frames argument: evaluate every frame in this episode.
python scripts/run_open_loop_inference_lerobot.py \
  --dataset "$DATASET" \
  --dataset_root "$DATASET_ROOT" \
  --episode_idx "$EPISODE_IDX" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --policy_type molmoact2 \
  --norm_tag panda_drawer_ee \
  --video_backend pyav \
  --device cuda \
  --seed 6198 \
  --verbose

echo "Completed full evaluation for episode ${EPISODE_IDX}"
echo "Summary: ${OUTPUT_DIR}/summary.json"
