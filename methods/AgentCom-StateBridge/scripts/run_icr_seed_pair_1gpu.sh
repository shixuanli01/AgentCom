#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:?usage: run_icr_seed_pair_1gpu.sh ARTIFACT_ROOT REPLICATION_ID}"
REPLICATION_ID="${2:?usage: run_icr_seed_pair_1gpu.sh ARTIFACT_ROOT REPLICATION_ID}"
PYTHON="${PYTHON:-/workspace/AgentCom/.venv-agentcom/bin/python}"
WORLD_SIZE="${WORKERS_PER_GPU:-2}"
CONDITIONS="none,true_text,self_text,other_text,true_statebridge,self_statebridge,other_statebridge"

mkdir -p "$ARTIFACT_ROOT"/{prebeliefs,messages,revisions,analysis,logs}

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
  [[ "$failed" -eq 0 ]]
}

echo "[$REPLICATION_ID] phase1: 300 items x 2 agents, logical_world=$WORLD_SIZE"
for ((rank=0; rank<WORLD_SIZE; rank++)); do
  LOCAL_RANK=0 "$PYTHON" -u -m icr.prebeliefs \
    --artifact-root "$ARTIFACT_ROOT" \
    --replication-id "$REPLICATION_ID" \
    --rank "$rank" --world-size "$WORLD_SIZE" \
    > "$ARTIFACT_ROOT/logs/prebelief_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
wait_for_phase
"$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
  --phase prebeliefs --require-complete
"$PYTHON" -m icr.verify --artifact-root "$ARTIFACT_ROOT" \
  > "$ARTIFACT_ROOT/logs/verification.log" 2>&1

echo "[$REPLICATION_ID] phase2: 600 directions x 7 conditions, logical_world=$WORLD_SIZE"
for ((rank=0; rank<WORLD_SIZE; rank++)); do
  LOCAL_RANK=0 "$PYTHON" -u -m icr.revisions \
    --artifact-root "$ARTIFACT_ROOT" \
    --conditions "$CONDITIONS" \
    --output-tag full --rank "$rank" --world-size "$WORLD_SIZE" \
    > "$ARTIFACT_ROOT/logs/revision_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
wait_for_phase
"$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
  --phase revisions --require-complete
"$PYTHON" -m icr.analysis --artifact-root "$ARTIFACT_ROOT"
"$PYTHON" -m icr.analysis_v2 \
  --source-root "$ARTIFACT_ROOT" --seed-root "$ARTIFACT_ROOT" \
  --num-bootstrap 10000 --bootstrap-seed 20260914
echo "[$REPLICATION_ID] complete"

trap - INT TERM EXIT
