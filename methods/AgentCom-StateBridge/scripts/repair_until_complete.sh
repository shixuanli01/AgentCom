#!/usr/bin/env bash
# Keep doubling a dataset's token budget until nothing truncates, or a ceiling
# is reached.
#
# One doubling is often not enough. Raising GSM8K from 2048 to 4096 took its
# truncated prebeliefs from 226 to 73 rather than to zero, because some items
# truncate from the model looping, not from a merely tight budget.
#
# Truncation is not uniform noise. A truncated generation never states its
# answer and is scored wrong, and "scored wrong" is exactly the condition for
# entering the disagreement and both-wrong subsets, which are the only places CR,
# PR, and SR can be measured. On GSM8K, 2.8% of prebeliefs being truncated meant
# 65.8% of the disagreement items carried one.
#
# Usage:
#   bash scripts/repair_until_complete.sh TASK [CEILING] [WORKERS_PER_GPU]
set -uo pipefail

TASK="${1:?usage: repair_until_complete.sh TASK [CEILING] [WORKERS_PER_GPU]}"
CEILING="${2:-32768}"
WORKERS="${3:-2}"

cd /workspace/AgentCom/methods/AgentCom-StateBridge
export PYTHONPATH=.
PY=/workspace/AgentCom/.venv-agentcom/bin/python
ROOT="artifacts/icr_v3/${TASK}_full_seed42"
LOG=artifacts/icr_v3/queue.log

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

truncated_count() {
  "$PY" - "$ROOT" <<'PYEOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
n = 0
for path in root.glob("prebeliefs/rank*/records/item_*.json"):
    if not json.loads(path.read_text(encoding="utf-8")).get("hit_eos", True):
        n += 1
print(n)
PYEOF
}

current_budget() {
  "$PY" -c "import json;print(json.load(open('$ROOT/config.json'))['generation']['max_new_tokens'])"
}

pass=0
while true; do
  remaining=$(truncated_count)
  budget=$(current_budget)
  log "$TASK: budget $budget, truncated prebeliefs $remaining"

  if [[ "$remaining" -eq 0 ]]; then
    log "$TASK: nothing truncated; done after $pass pass(es)"
    break
  fi
  if [[ "$budget" -ge "$CEILING" ]]; then
    log "$TASK: at ceiling $CEILING with $remaining still truncated; stopping"
    log "$TASK: report these as a separate category rather than as wrong answers"
    break
  fi

  pass=$((pass + 1))
  next=$((budget * 2))
  [[ "$next" -gt "$CEILING" ]] && next="$CEILING"
  log ">>> $TASK pass $pass: raising $budget -> $next"
  bash scripts/rerun_truncated.sh "$TASK" "$next" "$WORKERS" >> "$LOG" 2>&1
done

"$PY" scripts/phase1_yield.py artifacts/icr_v3/*_full_seed42 | tee -a "$LOG"
