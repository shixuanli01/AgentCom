#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-artifacts/tmr_v1}"
PYTHON="${PYTHON:-python}"
BASELINE="${BASELINE:-}"
GPUS=(0 1 2 3)
PIDS=()
RUN_DIR="$ROOT/medqa300_last64_seed42"

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

mkdir -p "$RUN_DIR"
for shard_index in 0 1 2 3 4 5 6 7; do
  gpu="${GPUS[$((shard_index % 4))]}"
  printf '[%s] starting last64 shard=%s/8 gpu=%s\n' \
    "$(date --iso-8601=seconds)" "$shard_index" "$gpu"
  PYTHONPATH=. "$PYTHON" -u -m agentcom.tmr_eval \
    --communication-method tmr \
    --tmr-selection last64 \
    --coverage-scope full \
    --gpu "$gpu" \
    --num-shards 8 \
    --shard-index "$shard_index" \
    "${baseline_args[@]}" \
    --run-dir "$RUN_DIR" \
    >"$RUN_DIR/last64_shard_${shard_index}.log" 2>&1 &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
PIDS=()

if ((failed)); then
  printf '[%s] one or more last64 shards failed; inspect shard logs\n' \
    "$(date --iso-8601=seconds)" >&2
  exit 1
fi

"$PYTHON" - "$RUN_DIR/summary.json" <<'PY'
import json
import sys

path = sys.argv[1]
summary = json.load(open(path, encoding="utf-8"))
if summary["status"] != "complete" or summary["completed_records"] != 300:
    raise SystemExit(f"incomplete run: {path}: {summary}")
PY

printf '[%s] eight-shard last64 run complete\n' "$(date --iso-8601=seconds)"
