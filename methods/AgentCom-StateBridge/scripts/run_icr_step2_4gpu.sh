#!/usr/bin/env bash
set -euo pipefail

BASE_ROOT="${1:-artifacts/icr_medqa300}"
PYTHON="${PYTHON:-/workspace/AgentCom/.venv-agentcom/bin/python}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
SEED_INDICES="${SEED_INDICES:-1 2 3 4}"
POOL_REPLICATIONS="${POOL_REPLICATIONS:-seed_pair_00 seed_pair_01 seed_pair_02 seed_pair_03 seed_pair_04}"

mkdir -p "$BASE_ROOT/step2_logs"
pids=()
terminate_children() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM EXIT

for index in $SEED_INDICES; do
  replication_id=$(printf 'seed_pair_%02d' "$index")
  gpu=$((index - 1))
  root="$BASE_ROOT/$replication_id"
  echo "[ICR Step2] GPU $gpu -> $replication_id; estimated generations=4800; workers=$WORKERS_PER_GPU"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHON="$PYTHON" WORKERS_PER_GPU="$WORKERS_PER_GPU" \
    bash scripts/run_icr_seed_pair_1gpu.sh "$root" "$replication_id" \
    > "$BASE_ROOT/step2_logs/${replication_id}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
pids=()
if [[ "$failed" -ne 0 ]]; then
  echo "[ICR Step2] at least one seed pair failed; rerun this command to resume" >&2
  exit 1
fi

"$PYTHON" -m icr.pooled_analysis \
  --base-root "$BASE_ROOT" \
  --replications $POOL_REPLICATIONS \
  --num-bootstrap 10000 --bootstrap-seed 20260915
echo "[ICR Step2] pooled five-seed analysis complete"

trap - INT TERM EXIT
