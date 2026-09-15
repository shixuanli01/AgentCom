#!/usr/bin/env bash
set -euo pipefail

BASE_ROOT="${1:-artifacts/icr_medqa300}"
PYTHON="${PYTHON:-/workspace/AgentCom/.venv-agentcom/bin/python}"
CONDITIONS="none,true_text,self_text,other_text,true_statebridge,self_statebridge,other_statebridge"

mkdir -p "$BASE_ROOT/rebalance_logs"
pids=()
terminate_children() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM EXIT

# Keep two model workers per GPU, as validated by the original run.  Each seed
# receives two GPUs and four logical shards; --global-resume safely reuses all
# records produced under the earlier two-shard topology.
for seed_index in 1 2; do
  replication_id=$(printf 'seed_pair_%02d' "$seed_index")
  root="$BASE_ROOT/$replication_id"
  primary_gpu=$((seed_index - 1))
  extra_gpu=$((seed_index + 1))
  for rank in 0 1 2 3; do
    if [[ "$rank" -lt 2 ]]; then
      gpu=$primary_gpu
    else
      gpu=$extra_gpu
    fi
    echo "[ICR rebalance] $replication_id rank=$rank/4 GPU=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m icr.revisions \
      --artifact-root "$root" --conditions "$CONDITIONS" \
      --rank "$rank" --world-size 4 --output-tag rebalance --global-resume \
      > "$BASE_ROOT/rebalance_logs/${replication_id}_rank${rank}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
pids=()
if [[ "$failed" -ne 0 ]]; then
  echo "[ICR rebalance] at least one shard failed; rerun to resume" >&2
  exit 1
fi

for seed_index in 1 2; do
  replication_id=$(printf 'seed_pair_%02d' "$seed_index")
  root="$BASE_ROOT/$replication_id"
  "$PYTHON" -m icr.merge --artifact-root "$root" \
    --phase revisions --require-complete
  "$PYTHON" -m icr.analysis --artifact-root "$root"
  "$PYTHON" -m icr.analysis_v2 \
    --source-root "$root" --seed-root "$root" \
    --num-bootstrap 10000 --bootstrap-seed 20260914
done

"$PYTHON" -m icr.pooled_analysis \
  --base-root "$BASE_ROOT" \
  --replications seed_pair_00 seed_pair_01 seed_pair_02 \
  --num-bootstrap 10000 --bootstrap-seed 20260915
echo "[ICR rebalance] both seeds and pooled three-seed analysis complete"

trap - INT TERM EXIT
