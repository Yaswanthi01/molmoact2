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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
fi
EXPERIMENTS_ROOT="$REPO_ROOT/experiments"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv_molmoact312}"
cd "$EXPERIMENTS_ROOT"
source "$VENV_PATH/bin/activate"

export LEROBOT_DATA_ROOT="${LEROBOT_DATA_ROOT:-$REPO_ROOT/datasets}"
export MOLMO_DATA_DIR="${MOLMO_DATA_DIR:-$REPO_ROOT/datasets}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}"

CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/lerobot_eval_single_config.env}"
if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

DATASET="${DATASET:-}"
CHECKPOINT="${CHECKPOINT:-}"
EPISODE_IDX="${EPISODE_IDX:-0}"
EVALUATION_ROOT="${EVALUATION_ROOT:-$REPO_ROOT/evaluations/${DATASET}-eval-single}"
POLICY_TYPE="${POLICY_TYPE:-molmoact2}"
NORM_TAG="${NORM_TAG:-}"
if [[ -z "$NORM_TAG" ]]; then
  NORM_TAG="$(basename "${DATASET##*/}")"
fi
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-6198}"
N_ACTION_STEPS="${N_ACTION_STEPS:-30}"
NUM_STEPS="${NUM_STEPS:-}"
SEQ_LEN="${SEQ_LEN:-}"
INFERENCE_ACTION_MODE="${INFERENCE_ACTION_MODE:-continuous}"
HF_CKPT="${HF_CKPT:-false}"
ENABLE_DEPTH_REASONING="${ENABLE_DEPTH_REASONING:-false}"
ENABLE_INFERENCE_CUDA_GRAPH="${ENABLE_INFERENCE_CUDA_GRAPH:-true}"
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
  --dataset_root "${DATASET_ROOT:-$LEROBOT_DATA_ROOT/$DATASET}"
  --episode_idx "$EPISODE_IDX"
  --checkpoint "$CHECKPOINT"
  --output_dir "$OUTPUT_DIR"
  --policy_type "$POLICY_TYPE"
  --norm_tag "$NORM_TAG"
  --video_backend "$VIDEO_BACKEND"
  --device "$DEVICE"
  --seed "$SEED"
  --inference_action_mode "$INFERENCE_ACTION_MODE"
)

if [[ -n "$N_ACTION_STEPS" ]]; then
  args+=(--n_action_steps "$N_ACTION_STEPS")
fi
if [[ -n "$NUM_STEPS" ]]; then
  args+=(--num_steps "$NUM_STEPS")
fi
if [[ -n "$SEQ_LEN" ]]; then
  args+=(--seq_len "$SEQ_LEN")
fi
if [[ "$HF_CKPT" == "true" || "$HF_CKPT" == "1" ]]; then
  args+=(--hf_ckpt true)
fi
if [[ "$ENABLE_DEPTH_REASONING" == "true" || "$ENABLE_DEPTH_REASONING" == "1" ]]; then
  args+=(--enable_depth_reasoning)
fi
if [[ "$ENABLE_INFERENCE_CUDA_GRAPH" == "false" || "$ENABLE_INFERENCE_CUDA_GRAPH" == "0" ]]; then
  args+=(--disable_inference_cuda_graph)
fi
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
