#!/bin/bash
#SBATCH --job-name=molmoact2-lerobot-eval
#SBATCH --account=ka
#SBATCH --partition=gpu_h100_short,gpu_a100_il,gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:28:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

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

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Python: $(which python)"
python --version
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

export LEROBOT_DATA_ROOT="${LEROBOT_DATA_ROOT:-$REPO_ROOT/datasets}"
export MOLMO_DATA_DIR="${MOLMO_DATA_DIR:-$REPO_ROOT/datasets}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}"

CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/lerobot_eval_config.env}"
if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

DATASET="${DATASET:-}"
if [[ -z "$DATASET" ]]; then
  echo "ERROR: DATASET must be set, e.g. panda_drawer_ee_molmoact2_v3" >&2
  exit 1
fi

DATASET_ROOT="${DATASET_ROOT:-$LEROBOT_DATA_ROOT/$DATASET}"
CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "$CHECKPOINT" ]]; then
  echo "ERROR: CHECKPOINT must be set" >&2
  exit 1
fi

EVALUATION_ROOT="${EVALUATION_ROOT:-$HOME/molmoact2/evaluations/${DATASET}-eval}"
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
EPISODE_IDX="${EPISODE_IDX:-${SLURM_ARRAY_TASK_ID:-0}}"
EPISODE_FRAMES="${EPISODE_FRAMES:-}"
VERBOSE="${VERBOSE:-false}"
DRY_RUN="${DRY_RUN:-false}"
RUN_REPORTS="${RUN_REPORTS:-true}"
EPISODE_IDS="${EPISODE_IDS:-}"
EPISODE_START="${EPISODE_START:-}"
EPISODE_END="${EPISODE_END:-}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-}"

declare -a episode_ids=()

if [[ -n "$EPISODE_IDS" ]]; then
  IFS=',' read -ra raw_episode_ids <<< "$EPISODE_IDS"
  for raw_id in "${raw_episode_ids[@]}"; do
    candidate="$(echo "$raw_id" | tr -d '[:space:]')"
    [[ -n "$candidate" ]] || continue
    episode_ids+=("$candidate")
  done
elif [[ -n "$EPISODE_START" && -n "$EPISODE_END" ]]; then
  start="$EPISODE_START"
  end="$EPISODE_END"
  if (( start > end )); then
    echo "ERROR: EPISODE_START ($start) must be <= EPISODE_END ($end)" >&2
    exit 1
  fi
  for ((episode = start; episode <= end; episode++)); do
    episode_ids+=("$episode")
  done
else
  episode_ids=("$EPISODE_IDX")
fi

if [[ ${#episode_ids[@]} -eq 0 ]]; then
  echo "ERROR: no episodes selected. Set EPISODE_IDS, EPISODE_START/EPISODE_END, or EPISODE_IDX." >&2
  exit 1
fi

expected_episode_count="${EXPECTED_EPISODES:-${#episode_ids[@]}}"

mkdir -p logs "$EVALUATION_ROOT" "$HF_HOME"

echo "DATASET=$DATASET"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "CHECKPOINT=$CHECKPOINT"
echo "EVALUATION_ROOT=$EVALUATION_ROOT"
echo "EPISODE_IDS=${episode_ids[*]}"
echo "POLICY_TYPE=$POLICY_TYPE"
echo "NORM_TAG=$NORM_TAG"
echo "VIDEO_BACKEND=$VIDEO_BACKEND"
echo "DEVICE=$DEVICE"
echo "SEED=$SEED"
echo "N_ACTION_STEPS=$N_ACTION_STEPS"
echo "NUM_STEPS=$NUM_STEPS"
echo "SEQ_LEN=$SEQ_LEN"
echo "INFERENCE_ACTION_MODE=$INFERENCE_ACTION_MODE"
echo "HF_CKPT=$HF_CKPT"
echo "ENABLE_DEPTH_REASONING=$ENABLE_DEPTH_REASONING"
echo "ENABLE_INFERENCE_CUDA_GRAPH=$ENABLE_INFERENCE_CUDA_GRAPH"
echo "EPISODE_FRAMES=$EPISODE_FRAMES"
echo "EXPECTED_EPISODES=$expected_episode_count"
echo "DRY_RUN=$DRY_RUN"
echo "RUN_REPORTS=$RUN_REPORTS"

for episode_idx in "${episode_ids[@]}"; do
  OUTPUT_DIR="$(printf '%s/episode%03d' "$EVALUATION_ROOT" "$episode_idx")"
  mkdir -p "$OUTPUT_DIR"

  default_args=(
    python scripts/run_open_loop_inference_lerobot.py
    --dataset "$DATASET"
    --dataset_root "$DATASET_ROOT"
    --episode_idx "$episode_idx"
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
    default_args+=(--n_action_steps "$N_ACTION_STEPS")
  fi

  if [[ -n "$NUM_STEPS" ]]; then
    default_args+=(--num_steps "$NUM_STEPS")
  fi

  if [[ -n "$SEQ_LEN" ]]; then
    default_args+=(--seq_len "$SEQ_LEN")
  fi

  if [[ "$HF_CKPT" == "true" || "$HF_CKPT" == "1" ]]; then
    default_args+=(--hf_ckpt true)
  fi

  if [[ "$ENABLE_DEPTH_REASONING" == "true" || "$ENABLE_DEPTH_REASONING" == "1" ]]; then
    default_args+=(--enable_depth_reasoning)
  fi

  if [[ "$ENABLE_INFERENCE_CUDA_GRAPH" == "false" || "$ENABLE_INFERENCE_CUDA_GRAPH" == "0" ]]; then
    default_args+=(--disable_inference_cuda_graph)
  fi

  if [[ "$VERBOSE" == "true" || "$VERBOSE" == "1" ]]; then
    default_args+=(--verbose)
  fi

  if [[ -n "$EPISODE_FRAMES" ]]; then
    default_args+=(--episode_frames "$EPISODE_FRAMES")
  fi

  printf 'Episode %s -> %s\n' "$episode_idx" "$OUTPUT_DIR"
  printf '%q ' "${default_args[@]}"
  echo

  if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] skipping execution"
  else
    "${default_args[@]}"
  fi
done

if [[ "$RUN_REPORTS" == "true" || "$RUN_REPORTS" == "1" ]]; then
  if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] would run aggregate reports"
  else
    python scripts/aggregate_open_loop_evaluations.py "$EVALUATION_ROOT" --expected-episodes "$expected_episode_count" --allow-incomplete
    python scripts/evaluate_gripper_binary.py "$EVALUATION_ROOT" --expected-episodes "$expected_episode_count" --allow-incomplete
    python scripts/build_final_evaluation_report.py "$EVALUATION_ROOT"
  fi
fi
