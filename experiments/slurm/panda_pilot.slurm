#!/bin/bash
#SBATCH --job-name=molmoact2-panda-ee-pilot
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
export WANDB_PROJECT="molmoact2-panda-ee-pilot"
export WANDB_ENTITY="local"
export WANDB_MODE="offline"

mkdir -p logs "$HOME/molmoact2/checkpoints" "$HF_HOME"

torchrun --standalone --nproc-per-node=1 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  panda_drawer_ee \
  --frame_loading_backend=av \
  --max_duration=100 \
  --log_interval=1 \
  --device_batch_size=1 \
  --global_batch_size=8 \
  --num_workers=0 \
  --pin_memory=false \
  --save_folder="$HOME/molmoact2/checkpoints/panda-ee-pilot" \
  --save_interval=50 \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none
