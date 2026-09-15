#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-artifacts/icr_medqa300_v2_seed42}"
PYTHON="${PYTHON:-python}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
GPU_COUNT=4
PREBELIEF_WORLD=$((GPU_COUNT * WORKERS_PER_GPU))
TEXT_WORLD=$((2 * WORKERS_PER_GPU))
STATEBRIDGE_WORLD=$((2 * WORKERS_PER_GPU))
TEXT_CONDITIONS="none,true_text,self_text,other_text"
STATEBRIDGE_CONDITIONS="true_statebridge,self_statebridge,other_statebridge"

mkdir -p "$ARTIFACT_ROOT/logs"

pids=()
terminate_children() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM EXIT

wait_for_phase() {
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  pids=()
  if [[ "$failed" -ne 0 ]]; then
    return 1
  fi
}

echo "[ICR formal] phase1 prebeliefs: world=$PREBELIEF_WORLD workers_per_gpu=$WORKERS_PER_GPU"
for ((rank=0; rank<PREBELIEF_WORLD; rank++)); do
  gpu=$((rank % GPU_COUNT))
  LOCAL_RANK=0 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m icr.prebeliefs \
    --artifact-root "$ARTIFACT_ROOT" \
    --rank "$rank" --world-size "$PREBELIEF_WORLD" \
    > "$ARTIFACT_ROOT/logs/prebelief_rank${rank}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
wait_for_phase
"$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
  --phase prebeliefs --require-complete
"$PYTHON" -m icr.verify --artifact-root "$ARTIFACT_ROOT"

echo "[ICR formal] phase2 text and StateBridge groups starting concurrently"
for ((rank=0; rank<TEXT_WORLD; rank++)); do
  gpu=$((rank % 2))
  LOCAL_RANK=0 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m icr.revisions \
    --artifact-root "$ARTIFACT_ROOT" \
    --conditions "$TEXT_CONDITIONS" \
    --output-tag text --rank "$rank" --world-size "$TEXT_WORLD" \
    > "$ARTIFACT_ROOT/logs/revision_text_rank${rank}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
for ((rank=0; rank<STATEBRIDGE_WORLD; rank++)); do
  gpu=$((2 + rank % 2))
  LOCAL_RANK=0 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m icr.revisions \
    --artifact-root "$ARTIFACT_ROOT" \
    --conditions "$STATEBRIDGE_CONDITIONS" \
    --output-tag statebridge --rank "$rank" --world-size "$STATEBRIDGE_WORLD" \
    > "$ARTIFACT_ROOT/logs/revision_statebridge_rank${rank}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
wait_for_phase

"$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
  --phase revisions --require-complete
"$PYTHON" -m icr.analysis --artifact-root "$ARTIFACT_ROOT"
echo "[ICR formal] complete: $ARTIFACT_ROOT"

trap - INT TERM EXIT
