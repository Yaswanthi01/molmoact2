#!/bin/bash
#SBATCH --job-name=molmoact2-lerobot-eval-single
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

export LEROBOT_DATA_ROOT="${LEROBOT_DATA_ROOT:-$HOME/molmoact2/datasets}"
export MOLMO_DATA_DIR="${MOLMO_DATA_DIR:-$HOME/molmoact2/datasets}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
export HF_HOME="${HF_HOME:-$HOME/molmoact2/.cache/huggingface}"
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}"

CONFIG_FILE="${CONFIG_FILE:-$HOME/molmoact2/experiments/slurm/lerobot_eval_single_config.env}"
if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

DATASET="${DATASET:-}"
CHECKPOINT="${CHECKPOINT:-}"
EPISODE_IDX="${EPISODE_IDX:-0}"
EVALUATION_ROOT="${EVALUATION_ROOT:-$HOME/molmoact2/evaluations/${DATASET}-eval-single}"
POLICY_TYPE="${POLICY_TYPE:-molmoact2}"
NORM_TAG="${NORM_TAG:-}"
if [[ -z "$NORM_TAG" ]]; then
  NORM_TAG="$(basename "${DATASET##*/}")"
fi
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-6198}"
EPISODE_FRAMES="${EPISODE_FRAMES:-}"
VERBOSE="${VERBOSE:-false}"

if [[ -z "$DATASET" ]]; then
  echo "ERROR: DATASET must be set" >&2
  exit 1
fi
if [[ -z "$CHECKPOINT" ]]; then
  echo "ERROR: CHECKPOINT must be set" >&2
  exit 1
fi

mkdir -p logs "$EVALUATION_ROOT" "$HF_HOME"
OUTPUT_DIR="$(printf '%s/episode%03d' "$EVALUATION_ROOT" "$EPISODE_IDX")"
mkdir -p "$OUTPUT_DIR"

args=(
  python scripts/run_open_loop_inference_lerobot.py
  --dataset "$DATASET"
  --dataset_root "$HOME/molmoact2/datasets/$DATASET"
  --episode_idx "$EPISODE_IDX"
  --checkpoint "$CHECKPOINT"
  --output_dir "$OUTPUT_DIR"
  --policy_type "$POLICY_TYPE"
  --norm_tag "$NORM_TAG"
  --video_backend "$VIDEO_BACKEND"
  --device "$DEVICE"
  --seed "$SEED"
)

if [[ "$VERBOSE" == "true" || "$VERBOSE" == "1" ]]; then
  args+=(--verbose)
fi
if [[ -n "$EPISODE_FRAMES" ]]; then
  args+=(--episode_frames "$EPISODE_FRAMES")
fi

printf 'Evaluating single episode %s\n' "$EPISODE_IDX"
printf '%q ' "${args[@]}"
echo
"${args[@]}"
