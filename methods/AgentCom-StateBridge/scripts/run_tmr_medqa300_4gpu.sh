#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-artifacts/tmr_v1}"
PYTHON="${PYTHON:-python}"
BASELINE="${BASELINE:-}"
GPUS=(0 1 2 3)
PIDS=()

baseline_args=()
if [[ -n "$BASELINE" ]]; then
  baseline_args=(--baseline-summary "$BASELINE")
fi

stop_workers() {
  if ((${#PIDS[@]})); then
    kill -TERM "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap stop_workers INT TERM

launch_variant() {
  local selection="$1"
  local run_dir="$ROOT/medqa300_${selection}_seed42"
  mkdir -p "$run_dir"
  for shard_index in 0 1 2 3; do
    local gpu="${GPUS[$shard_index]}"
    printf '[%s] starting %s shard=%s/4 gpu=%s\n' \
      "$(date --iso-8601=seconds)" "$selection" "$shard_index" "$gpu"
    PYTHONPATH=. "$PYTHON" -u -m agentcom.tmr_eval \
      --communication-method tmr \
      --tmr-selection "$selection" \
      --coverage-scope full \
      --gpu "$gpu" \
      --num-shards 4 \
      --shard-index "$shard_index" \
      "${baseline_args[@]}" \
      --run-dir "$run_dir" \
      >"$run_dir/shard_${shard_index}.log" 2>&1 &
    PIDS+=("$!")
  done
}

launch_variant last64
launch_variant coverage64

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
PIDS=()

if ((failed)); then
  printf '[%s] one or more TMR shards failed; inspect shard logs\n' \
    "$(date --iso-8601=seconds)" >&2
  exit 1
fi

for selection in last64 coverage64; do
  summary="$ROOT/medqa300_${selection}_seed42/summary.json"
  "$PYTHON" - "$summary" <<'PY'
import json
import sys

path = sys.argv[1]
summary = json.load(open(path, encoding="utf-8"))
if summary["status"] != "complete" or summary["completed_records"] != 300:
    raise SystemExit(f"incomplete run: {path}: {summary}")
PY
done

printf '[%s] four-GPU TMR comparison complete\n' "$(date --iso-8601=seconds)"
