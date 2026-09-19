#!/usr/bin/env bash
# Append Evidence Channel V1 to a completed ICR-V3 baseline artifact.
#
# Usage:
#   bash scripts/run_v3_evidence_channel.sh ARTIFACT_ROOT
#
# This launcher never generates prebeliefs or reruns baseline conditions. It
# requires the exact completed V3 baseline artifact so every comparison shares
# the same A/B trajectories and directional revision seeds.
set -euo pipefail

ARTIFACT_ROOT="${1:?usage: run_v3_evidence_channel.sh ARTIFACT_ROOT}"
DEFAULT_PYTHON=/workspace/AgentCom/.venv-agentcom/bin/python
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=python
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
read -r -a GPUS <<< "${CUDA_DEVICES:-0 1 2 3}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
WORLD=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))

mkdir -p "$ARTIFACT_ROOT/logs"

"$PYTHON" - "$ARTIFACT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
required = {"none", "true_text", "true_statebridge", "true_latentmas"}
completed = set(config.get("completed_revision_conditions") or ())
checks = {
    "protocol": config.get("protocol") == "ICR-V3",
    "prompt": config.get("revision_prompt_version") == "icr_v3_mid_injection",
    "task": config.get("dataset") == "medqa",
    "items": len(config.get("selected_item_ids") or ()) == 300,
    "replication": config.get("replication_id") == "seed_pair_00",
    "baseline_conditions": required <= completed,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Evidence V1 prerequisite failed: {failed}; config={config}")
print("Evidence V1 prerequisites passed; reusing frozen V3 prebeliefs and baselines")
PY

pids=()
terminate_children() {
  local pid
  for pid in "${pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
trap terminate_children INT TERM EXIT

rank=0
for gpu in "${GPUS[@]}"; do
  for ((worker = 0; worker < WORKERS_PER_GPU; worker++)); do
    CUDA_VISIBLE_DEVICES="$gpu" LOCAL_RANK=0 "$PYTHON" -u -m icr.revisions \
      --artifact-root "$ARTIFACT_ROOT" \
      --conditions true_evidence \
      --global-resume --output-tag evidence_v1 \
      --rank "$rank" --world-size "$WORLD" \
      > "$ARTIFACT_ROOT/logs/evidence_v1_rank${rank}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
    rank=$((rank + 1))
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
pids=()
[[ "$failed" -eq 0 ]]

"$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
  --phase revisions --require-complete
"$PYTHON" -m icr.analysis --artifact-root "$ARTIFACT_ROOT" \
  > "$ARTIFACT_ROOT/logs/analysis_with_evidence.log" 2>&1
"$PYTHON" -m icr.analysis_v2 \
  --source-root "$ARTIFACT_ROOT" --seed-root "$ARTIFACT_ROOT" \
  --num-bootstrap 10000 --bootstrap-seed 20260919 \
  > "$ARTIFACT_ROOT/logs/analysis_v2_with_evidence.log" 2>&1

trap - INT TERM EXIT
echo "Evidence Channel V1 complete: $ARTIFACT_ROOT/analysis_v2/summary.md"
