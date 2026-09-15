#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-artifacts/icr_medqa300_v2_sourcefront_seed42}"
PYTHON="${PYTHON:-/workspace/AgentCom/.venv-agentcom/bin/python}"
LATENT_STEPS="${LATENT_STEPS:-10}"

mkdir -p "$ARTIFACT_ROOT/logs"
pids=()
terminate_children() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM EXIT

for rank in 0 1; do
  gpu=$((rank + 2))
  echo "[ICR LatentMAS full] GPU $gpu rank=$rank"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m icr.revisions \
    --artifact-root "$ARTIFACT_ROOT" \
    --conditions true_latentmas,self_latentmas,other_latentmas \
    --rank "$rank" --world-size 2 \
    --output-tag latentmas --latent-steps "$LATENT_STEPS" \
    > "$ARTIFACT_ROOT/logs/latentmas_full_rank${rank}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
pids=()
trap - INT TERM EXIT
if [[ "$failed" -ne 0 ]]; then
  echo "[ICR LatentMAS full] at least one rank failed; rerun to resume" >&2
  exit 1
fi
echo "[ICR LatentMAS full] complete"
