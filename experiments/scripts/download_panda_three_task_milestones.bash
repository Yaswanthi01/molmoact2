#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-ka_ugqju@uc3-login1.scc.kit.edu}"
SOCK="${SOCK:-/tmp/panda-kit-three-task-${USER}.sock}"
RAW_ROOT="${RAW_ROOT:-/tmp/molmoact2-three-task-checkpoints/raw}"

exec python "$SCRIPT_DIR/download_panda_checkpoints.py" \
  --start 49 \
  --end 54 \
  --host "$REMOTE_HOST" \
  --socket "$SOCK" \
  --raw-root "$RAW_ROOT" \
  --method rsync
