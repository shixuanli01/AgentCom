#!/usr/bin/env bash
# Repair one dataset's truncated records at a raised token budget.
#
# max_new_tokens decides when generation stops, not how it samples, so records
# that reached EOS are exactly what the same seeds would produce under a larger
# budget and are kept. Only the truncated ones are regenerated.
#
# A regenerated prebelief invalidates more than itself: the sender message, the
# receiver's prior, and the pair classification of that item all change, so every
# revision record for that item is deleted and regenerated too.
#
# Usage:
#   bash scripts/rerun_truncated.sh TASK NEW_MAX_NEW_TOKENS [WORKERS_PER_GPU]
set -euo pipefail

TASK="${1:?usage: rerun_truncated.sh TASK NEW_MAX_NEW_TOKENS [WORKERS_PER_GPU]}"
BUDGET="${2:?missing NEW_MAX_NEW_TOKENS}"
WORKERS="${3:-2}"

cd /workspace/AgentCom/methods/AgentCom-StateBridge
export HF_HOME=/workspace/.hf_home
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/workspace/AgentCom/.venv-agentcom/bin/python
ROOT="artifacts/icr_v3/${TASK}_full_seed42"
LOG=artifacts/icr_v3/queue.log
CONDITIONS="none,true_text,true_statebridge,true_latentmas"
read -r -a GPUS <<< "${CUDA_DEVICES:-0 1 2 3}"
WORLD=$(( ${#GPUS[@]} * WORKERS ))

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

log "=== $TASK: raising budget to $BUDGET and repairing truncated records ==="

# Drop revisions that were derived from a prebelief about to change.
"$PY" - "$ROOT" <<'PYEOF'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
stale_items = set()
for path in root.glob("prebeliefs/rank*/records/item_*.json"):
    row = json.loads(path.read_text(encoding="utf-8"))
    if not row.get("hit_eos", True):
        stale_items.add(int(row["item_id"]))

removed = 0
for path in root.glob("revisions/rank*/records/item_*.json"):
    row = json.loads(path.read_text(encoding="utf-8"))
    if int(row["item_id"]) in stale_items:
        path.unlink()
        removed += 1
print(f"truncated prebeliefs: {len(stale_items)} items; stale revisions removed: {removed}")
PYEOF

"$PY" scripts/raise_token_budget.py "$ROOT" "$BUDGET"

launch() {
  local module="$1"; shift
  local rank=0 gpu worker pids=()
  for gpu in "${GPUS[@]}"; do
    for ((worker = 0; worker < WORKERS; worker++)); do
      CUDA_VISIBLE_DEVICES="$gpu" LOCAL_RANK=0 "$PY" -u -m "$module" \
        --artifact-root "$ROOT" --rank "$rank" --world-size "$WORLD" \
        --rerun-truncated "$@" \
        >> "$ROOT/logs/${module##*.}_repair_rank${rank}_gpu${gpu}.log" 2>&1 &
      pids+=("$!")
      rank=$((rank + 1))
    done
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}

log "$TASK: regenerating truncated prebeliefs on $WORLD workers"
launch icr.prebeliefs --task "$TASK" --replication-id seed_pair_00 || \
  log "$TASK: some prebelief workers failed; rerun resumes"
"$PY" -m icr.merge --artifact-root "$ROOT" --phase prebeliefs --require-complete >> "$LOG" 2>&1

log "$TASK: regenerating revisions"
launch icr.revisions --conditions "$CONDITIONS" --latent-steps 10 \
  --both-correct-sample 10 --global-resume --output-tag v3 || \
  log "$TASK: some revision workers failed; rerun resumes"

if "$PY" -m icr.merge --artifact-root "$ROOT" --phase revisions --require-complete >> "$LOG" 2>&1; then
  "$PY" -m icr.analysis --artifact-root "$ROOT" > "$ROOT/logs/analysis.log" 2>&1
  "$PY" -m icr.analysis_v2 --source-root "$ROOT" --seed-root "$ROOT" \
    --num-bootstrap 10000 --bootstrap-seed 20260919 > "$ROOT/logs/analysis_v2.log" 2>&1
  log "<<< $TASK repaired and analyzed"
else
  log "<<< $TASK still incomplete; rerun resumes"
fi

"$PY" scripts/phase1_yield.py artifacts/icr_v3/*_full_seed42 | tee -a "$LOG"
