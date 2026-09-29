#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-ka_ugqju@uc3-login1.scc.kit.edu}"
SOCK="${SOCK:-/tmp/panda-kit-three-task-${USER}.sock}"
RAW_ROOT="${RAW_ROOT:-/tmp/molmoact2-three-task-checkpoints/raw}"
HF_ROOT="${HF_ROOT:-/tmp/molmoact2-three-task-checkpoints/hf}"

extra_args=()
if [[ "${DELETE_RAW:-0}" == "1" ]]; then
  extra_args+=(--delete-raw)
fi

exec python "$SCRIPT_DIR/upload_panda_hf_to_cluster.py" \
  --start 49 \
  --end 54 \
  --host "$REMOTE_HOST" \
  --socket "$SOCK" \
  --raw-root "$RAW_ROOT" \
  --hf-root "$HF_ROOT" \
  "${extra_args[@]}"
