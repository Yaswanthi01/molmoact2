#!/bin/bash
#SBATCH --job-name=molmoact2-panda-ee-eval
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

CHECKPOINT="$HOME/molmoact2/checkpoints/panda-ee-pilot/step100"
OUTPUT_DIR="$HOME/molmoact2/evaluations/panda-ee-pilot-step100-episode0"

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Available checkpoints:" >&2
  find "$HOME/molmoact2/checkpoints/panda-ee-pilot" -maxdepth 1 -type d -name 'step*' -print >&2 || true
  exit 1
fi

mkdir -p logs "$OUTPUT_DIR" "$HF_HOME"

python scripts/run_open_loop_inference_lerobot.py \
  --dataset panda_drawer_ee_molmoact2_v3 \
  --dataset_root "$HOME/molmoact2/datasets/panda_drawer_ee_molmoact2_v3" \
  --episode_idx 0 \
  --episode_frames 0,10 \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --policy_type molmoact2 \
  --norm_tag panda_drawer_ee \
  --video_backend pyav \
  --device cuda \
  --verbose
