#!/bin/bash
#SBATCH --job-name=molmoact2-panda-milestone-eval
#SBATCH --account=ka
#SBATCH --partition=gpu_h100_short,gpu_a100_il,gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:28:00
#SBATCH --array=0-124%10
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "$HOME/molmoact2/experiments"
source "$HOME/molmoact2/.venv_molmoact312/bin/activate"

# Five checkpoints x 25 fixed episodes = 125 array tasks.
CHECKPOINT_STEPS=(40000 50000 60000 70000 75000)
EPISODE_INDICES=(
  0 4 8 12 17
  21 25 29 33 38
  42 46 50 54 58
  62 67 71 75 79
  83 88 92 96 100
)

NUM_EPISODES="${#EPISODE_INDICES[@]}"
CHECKPOINT_POSITION=$((SLURM_ARRAY_TASK_ID / NUM_EPISODES))
EPISODE_POSITION=$((SLURM_ARRAY_TASK_ID % NUM_EPISODES))
CHECKPOINT_STEP="${CHECKPOINT_STEPS[$CHECKPOINT_POSITION]}"
EPISODE_IDX="${EPISODE_INDICES[$EPISODE_POSITION]}"

CHECKPOINT="$HOME/molmoact2/checkpoints/panda-ee-50k-milestones/step${CHECKPOINT_STEP}"

DATASET="panda_drawer_ee_molmoact2_v3"
DATASET_ROOT="$HOME/molmoact2/datasets/$DATASET"
EVALUATION_ROOT="$HOME/molmoact2/evaluations/panda-ee-milestone-comparison/step${CHECKPOINT_STEP}"
OUTPUT_DIR="$(printf '%s/episode%03d' "$EVALUATION_ROOT" "$EPISODE_IDX")"

echo "Host: $(hostname)"
echo "Array job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Checkpoint: $CHECKPOINT"
echo "Episode: $EPISODE_IDX"
echo "Python: $(which python)"
python --version
nvidia-smi

export LEROBOT_DATA_ROOT="$HOME/molmoact2/datasets"
export MOLMO_DATA_DIR="$HOME/molmoact2/datasets"
export LEROBOT_VIDEO_BACKEND=pyav
export HF_HOME="$HOME/molmoact2/.cache/huggingface"
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}"

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Dataset not found: $DATASET_ROOT" >&2
  exit 1
fi

mkdir -p logs "$OUTPUT_DIR" "$HF_HOME"

# No --episode_frames argument: evaluate every frame in the selected episode.
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

echo "Completed checkpoint step${CHECKPOINT_STEP}, episode ${EPISODE_IDX}"
echo "Summary: ${OUTPUT_DIR}/summary.json"
