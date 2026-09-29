#!/usr/bin/env bash

set -uo pipefail

TARGET_STEPS="${TARGET_STEPS:-120000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
RETRY_SECONDS="${RETRY_SECONDS:-60}"
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-3}"
JOB_NAME="molmoact2-panda-three-window-full-120k-b64"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SLURM_SCRIPT="$REPO_ROOT/experiments/slurm/panda_three_task_window_proportional_full_120k_bs64.slurm"
MOLMO_WORKSPACE="${MOLMO_WORKSPACE:-$(ws_find molmoact2-checkpoints)}"
CHECKPOINT_DIR="$MOLMO_WORKSPACE/checkpoints/panda-three-task-window-proportional-full-h30s30-bs64-4gpu-120k"
LOG_ROOT="$MOLMO_WORKSPACE/logs/training/panda-three-task-weighting"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Slurm script not found: $SLURM_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$MOLMO_WORKSPACE" || ! -w "$MOLMO_WORKSPACE" ]]; then
  echo "Workspace is unavailable or not writable: $MOLMO_WORKSPACE" >&2
  exit 1
fi

latest_step() {
  local latest=0 checkpoint step
  for checkpoint in "$CHECKPOINT_DIR"/step*; do
    [[ -d "$checkpoint" ]] || continue
    step="${checkpoint##*/step}"
    if [[ "$step" =~ ^[0-9]+$ ]] && (( step > latest )); then
      latest="$step"
    fi
  done
  printf '%s\n' "$latest"
}

active_job() {
  squeue -h -u "$USER" -n "$JOB_NAME" -o '%A' | head -n 1
}

job_state() {
  local state
  state="$(sacct -X -n -P -j "$1" -o State 2>/dev/null | head -n 1 | cut -d '|' -f 1)"
  printf '%s\n' "${state%%+*}"
}

reported_step() {
  local job_id="$1" log_file="" step=""
  log_file="$(find "$LOG_ROOT" -type f -name "*-${job_id}.out" -print -quit 2>/dev/null)"
  if [[ -n "$log_file" ]]; then
    step="$(grep -oE '\[step=[0-9]+/' "$log_file" 2>/dev/null | tail -n 1 | tr -cd '0-9')"
  fi
  if [[ -z "$step" ]]; then
    step="$(latest_step)"
  fi
  printf '%s\n' "$step"
}

echo "Watching window-proportional (alpha=1) training until step${TARGET_STEPS}."
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Press Ctrl-C to stop the watcher; an active Slurm job will continue."

consecutive_failures=0
while true; do
  step="$(latest_step)"
  if (( step >= TARGET_STEPS )); then
    echo "Training target reached at step${step}."
    exit 0
  fi

  job_id="$(active_job)"
  if [[ -z "$job_id" ]]; then
    if ! job_id="$(sbatch --parsable "$SLURM_SCRIPT")"; then
      echo "Submission failed; retrying in ${RETRY_SECONDS} seconds." >&2
      sleep "$RETRY_SECONDS"
      continue
    fi
    job_id="${job_id%%;*}"
    echo "Submitted alpha=1 segment as job ${job_id} from step${step}."
  else
    echo "Watching existing alpha=1 job ${job_id} from step${step}."
  fi

  environment_retry=0
  while true; do
    queue_status="$(squeue -h -j "$job_id" -o '%T | elapsed %M | start %S | %R')"
    [[ -n "$queue_status" ]] || break
    if [[ "${queue_status,,}" == *"user env retrieval failed"* ]]; then
      printf '\n'
      echo "Job ${job_id} has an environment-retrieval hold; cancelling it for a clean retry."
      scancel "$job_id"
      environment_retry=1
      while squeue -h -j "$job_id" | grep -q .; do
        sleep 2
      done
      break
    fi
    step="$(reported_step "$job_id")"
    printf '\r[%s] job %s | %s | last logged step %s\033[K' \
      "$(date '+%F %T')" "$job_id" "$queue_status" "$step"
    sleep "$POLL_SECONDS"
  done
  printf '\n'

  state="$(job_state "$job_id")"
  step="$(latest_step)"
  echo "Job ${job_id} ended with state ${state:-UNKNOWN}; latest checkpoint is step${step}."

  if [[ "$state" == "COMPLETED" ]]; then
    consecutive_failures=0
  else
    ((consecutive_failures += 1))
    if (( consecutive_failures >= MAX_CONSECUTIVE_FAILURES )); then
      echo "Stopping after ${consecutive_failures} consecutive unsuccessful jobs." >&2
      exit 1
    fi
    if (( environment_retry == 1 )); then
      echo "Retrying after the environment-retrieval failure in ${RETRY_SECONDS} seconds."
    fi
  fi

  sleep "$RETRY_SECONDS"
done
