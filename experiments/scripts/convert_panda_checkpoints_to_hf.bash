#!/bin/bash

set -euo pipefail

MOLMO_WORKSPACE="${MOLMO_WORKSPACE:-$(ws_find molmoact2-checkpoints)}"
if [[ "$MOLMO_WORKSPACE" != /* || ! -d "$MOLMO_WORKSPACE" || ! -w "$MOLMO_WORKSPACE" ]]; then
  echo "MOLMO_WORKSPACE must be an absolute, existing, writable directory: $MOLMO_WORKSPACE" >&2
  exit 1
fi

TASKS=()
MODES=()
SOURCE_RUNS=()
STEPS=()
SOURCE_SUFFIXES=()

add_group() {
  local task="$1"
  local mode="$2"
  local source_runs="$3"
  local source_suffix="$4"
  shift 4
  local step
  for step in "$@"; do
    TASKS+=("$task")
    MODES+=("$mode")
    SOURCE_RUNS+=("$source_runs")
    STEPS+=("$step")
    SOURCE_SUFFIXES+=("$source_suffix")
  done
}

DRAWER_AND_CUP_STEPS=(10000 20000 30000 50000 60000 70000 80000 90000)

add_group \
  drawer full \
  "panda-drawer-ee-state-absolute-full-h30s30-bs32-2gpu-120k|panda-drawer-ee-state-absolute-full-h30s30-bs32-120k" \
  "" \
  "${DRAWER_AND_CUP_STEPS[@]}"
add_group \
  drawer lora \
  "panda-drawer-ee-state-absolute-lora-h30s30-bs32-120k" \
  "-merged" \
  "${DRAWER_AND_CUP_STEPS[@]}"
add_group \
  cup full \
  "panda-stack-two-cup-ee-state-absolute-full-h30s30-bs32-2gpu-120k|panda-stack-two-cup-ee-state-absolute-full-h30s30-bs32-120k" \
  "" \
  "${DRAWER_AND_CUP_STEPS[@]}"
add_group \
  cup lora \
  "panda-stack-two-cup-ee-state-absolute-lora-h30s30-bs32-120k" \
  "-merged" \
  "${DRAWER_AND_CUP_STEPS[@]}"
add_group \
  sort full \
  "panda-sort-three-absolute-all-full-h30s30-bs64-4gpu-120k|panda-sort-three-absolute-all-full-h30s30-bs64-120k" \
  "" \
  10000 20000 30000 40000 50000 60000
add_group \
  sort lora \
  "panda-sort-three-absolute-all-lora-action-h30s30-bs64-120k" \
  "-merged" \
  10000 20000 30000 40000

if (( ${#TASKS[@]} != 42 )); then
  echo "Internal error: expected 42 exports, found ${#TASKS[@]}" >&2
  exit 1
fi

resolve_source() {
  local index="$1"
  local checkpoint_name="step${STEPS[$index]}${SOURCE_SUFFIXES[$index]}"
  local source_run
  local candidate
  local run_candidates
  IFS='|' read -r -a run_candidates <<< "${SOURCE_RUNS[$index]}"
  for source_run in "${run_candidates[@]}"; do
    for candidate in \
      "$MOLMO_WORKSPACE/checkpoints/${source_run}-milestones/$checkpoint_name" \
      "$MOLMO_WORKSPACE/checkpoints/${source_run}/$checkpoint_name"; do
      if [[ -d "$candidate" ]]; then
        RESOLVED_SOURCE="$candidate"
        return 0
      fi
    done
  done
  return 1
}

print_item() {
  local index="$1"
  local checkpoint_name="step${STEPS[$index]}${SOURCE_SUFFIXES[$index]}"
  local output_dir="$MOLMO_WORKSPACE/hf_checkpoints/${TASKS[$index]}/${MODES[$index]}/step${STEPS[$index]}"
  printf '%02d  %-6s  %-4s  %-18s -> %s\n' \
    "$index" "${TASKS[$index]}" "${MODES[$index]}" "$checkpoint_name" "$output_dir"
}

ACTION="${1:---all}"
if [[ "$ACTION" == "--list" ]]; then
  for index in "${!TASKS[@]}"; do
    print_item "$index"
  done
  echo "Total exports: ${#TASKS[@]}"
  exit 0
fi

if [[ "$ACTION" == "--check" ]]; then
  missing=0
  for index in "${!TASKS[@]}"; do
    if resolve_source "$index"; then
      printf 'OK       %02d  %s\n' "$index" "$RESOLVED_SOURCE"
    else
      printf 'MISSING  %02d  %s/%s/step%s%s\n' \
        "$index" "${TASKS[$index]}" "${MODES[$index]}" \
        "${STEPS[$index]}" "${SOURCE_SUFFIXES[$index]}"
      missing=$((missing + 1))
    fi
  done
  if (( missing != 0 )); then
    echo "Missing $missing of ${#TASKS[@]} source checkpoints." >&2
    exit 1
  fi
  echo "All ${#TASKS[@]} source checkpoints are available."
  exit 0
fi

LOG_DIR="$MOLMO_WORKSPACE/logs/hf_conversion"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXPERIMENTS_DIR/.." && pwd)"
cd "$EXPERIMENTS_DIR"
source "$REPO_ROOT/.venv_molmoact312/bin/activate"

convert_one() {
  local index="$1"
  local output_dir
  local complete_marker
  local log_file

  if ! resolve_source "$index"; then
    echo "Source checkpoint is missing for conversion $index:" >&2
    print_item "$index" >&2
    return 1
  fi

  output_dir="$MOLMO_WORKSPACE/hf_checkpoints/${TASKS[$index]}/${MODES[$index]}/step${STEPS[$index]}"
  complete_marker="$output_dir/conversion_complete.json"
  log_file="$LOG_DIR/${TASKS[$index]}-${MODES[$index]}-step${STEPS[$index]}.log"

  if [[ -f "$output_dir/config.json" && -f "$output_dir/processor_config.json" && -f "$complete_marker" ]]; then
    echo "Already complete, skipping: $output_dir"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    echo "Incomplete output already exists: $output_dir" >&2
    echo "Move or delete only that incomplete directory, then retry conversion $index." >&2
    return 1
  fi

  mkdir -p "$(dirname "$output_dir")"
  echo "Converting $((index + 1))/${#TASKS[@]}"
  echo "Source: $RESOLVED_SOURCE"
  echo "Output: $output_dir"

  python -m olmo.hf_model.convert_molmoact2_to_hf \
    "$RESOLVED_SOURCE" \
    "$output_dir" \
    --attn_implementation sdpa \
    --max_shard_size 5GB \
    2>&1 | tee -a "$log_file"

  printf '{\n  "index": %s,\n  "task": "%s",\n  "mode": "%s",\n  "step": %s,\n  "source_checkpoint": "%s",\n  "output_directory": "%s"\n}\n' \
    "$index" "${TASKS[$index]}" "${MODES[$index]}" "${STEPS[$index]}" \
    "$RESOLVED_SOURCE" "$output_dir" > "$complete_marker"
  echo "Completed: $output_dir"
}

if [[ "$ACTION" == "--index" ]]; then
  INDEX="${2:-}"
  if [[ ! "$INDEX" =~ ^[0-9]+$ ]] || (( INDEX >= ${#TASKS[@]} )); then
    echo "Usage: bash $0 --index <0-$(( ${#TASKS[@]} - 1 ))>" >&2
    exit 1
  fi
  convert_one "$INDEX"
  exit 0
fi

if [[ "$ACTION" != "--all" ]]; then
  echo "Usage: bash $0 [--all|--list|--check|--index N]" >&2
  exit 1
fi

for index in "${!TASKS[@]}"; do
  convert_one "$index"
done

echo "All ${#TASKS[@]} conversions are complete."
