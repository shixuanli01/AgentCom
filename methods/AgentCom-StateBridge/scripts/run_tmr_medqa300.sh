#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-artifacts/tmr_v1}"
PYTHON="${PYTHON:-python}"
BASELINE="${BASELINE:-}"

run_variant() {
  local selection="$1"
  local run_dir="$ROOT/medqa300_${selection}_seed42"
  mkdir -p "$run_dir"
  printf '[%s] starting %s\n' "$(date --iso-8601=seconds)" "$selection"
  local baseline_args=()
  if [[ -n "$BASELINE" ]]; then
    baseline_args=(--baseline-summary "$BASELINE")
  fi
  PYTHONPATH=. "$PYTHON" -u -m agentcom.tmr_eval \
    --communication-method tmr \
    --tmr-selection "$selection" \
    --coverage-scope full \
    "${baseline_args[@]}" \
    --run-dir "$run_dir"
  local status
  status=$("$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$run_dir/summary.json")
  if [[ "$status" != "complete" ]]; then
    printf '[%s] stopped %s status=%s\n' \
      "$(date --iso-8601=seconds)" "$selection" "$status"
    exit 2
  fi
}

run_variant last64
run_variant coverage64
printf '[%s] TMR queue complete\n' "$(date --iso-8601=seconds)"
