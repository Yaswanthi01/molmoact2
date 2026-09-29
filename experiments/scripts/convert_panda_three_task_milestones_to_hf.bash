#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_ROOT="${RAW_ROOT:-/tmp/molmoact2-three-task-checkpoints/raw}"
HF_ROOT="${HF_ROOT:-/tmp/molmoact2-three-task-checkpoints/hf}"

exec python "$SCRIPT_DIR/convert_downloaded_panda_to_hf.py" \
  --start 49 \
  --end 54 \
  --raw-root "$RAW_ROOT" \
  --hf-root "$HF_ROOT"
