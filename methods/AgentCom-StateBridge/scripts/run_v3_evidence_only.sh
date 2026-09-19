#!/usr/bin/env bash
# Generate V3 prebeliefs and evaluate only Evidence Channel V1.
# Baseline conditions are intentionally not run by this launcher.
set -euo pipefail

ARTIFACT_ROOT="${1:-artifacts/icr_v3/medqa300_seed_pair_00_evidence_v1}"
PYTHON="${PYTHON:-python}"
GPU="${CUDA_DEVICE:-0}"
WORKERS="${WORKERS_PER_GPU:-2}"
LIMIT="${MEDQA_LIMIT:-300}"
REPLICATION_ID="${REPLICATION_ID:-seed_pair_00}"

mkdir -p "$ARTIFACT_ROOT/logs"
pids=()

terminate_children() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM EXIT

wait_for_workers() {
  local failed=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  pids=()
  [[ "$failed" -eq 0 ]]
}

launch() {
  local phase="$1"
  shift
  local rank
  for ((rank = 0; rank < WORKERS; rank++)); do
    CUDA_VISIBLE_DEVICES="$GPU" LOCAL_RANK=0 "$PYTHON" -u -m "$phase" \
      --artifact-root "$ARTIFACT_ROOT" \
      --rank "$rank" --world-size "$WORKERS" "$@" \
      > "$ARTIFACT_ROOT/logs/${phase##*.}_rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  wait_for_workers
}

echo "[Evidence V1] root=$ARTIFACT_ROOT limit=$LIMIT workers=$WORKERS gpu=$GPU"
if [[ -f "$ARTIFACT_ROOT/config.json" ]] && \
  "$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
    --phase prebeliefs --require-complete \
    > "$ARTIFACT_ROOT/logs/prebelief_cache_check.log" 2>&1; then
  echo "[Evidence V1] complete V3 prebelief cache found"
else
  launch icr.prebeliefs --task medqa --limit "$LIMIT" \
    --replication-id "$REPLICATION_ID"
  "$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
    --phase prebeliefs --require-complete
fi

if (( LIMIT >= 2 )); then
  "$PYTHON" -m icr.verify --artifact-root "$ARTIFACT_ROOT" \
    > "$ARTIFACT_ROOT/logs/verification.log" 2>&1
else
  # The shared verifier constructs an unrelated-item control and therefore
  # deliberately rejects a one-item cache. Evidence-only smoke runs need only
  # verify the two independent source records and their cached prefixes.
  "$PYTHON" - "$ARTIFACT_ROOT" > "$ARTIFACT_ROOT/logs/verification.log" <<'PY'
import json
import sys
from pathlib import Path

from safetensors.torch import load_file

root = Path(sys.argv[1])
rows = [
    json.loads(line)
    for line in (root / "prebeliefs" / "merged.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
assert len(rows) == 2
assert {row["agent_id"] for row in rows} == {"A", "B"}
assert len({row["generation_seed"] for row in rows}) == 2
assert len({row["prompt_sha256"] for row in rows}) == 1
for row in rows:
    prefix = load_file(str(root / row["statebridge_prefix_file"]), device="cpu")
    assert prefix["statebridge_prefix"].shape[1] == 64
print("single-item evidence-only verification: PASS")
PY
fi

launch icr.revisions --conditions true_evidence --global-resume \
  --output-tag evidence_only_v1
"$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
  --phase revisions --require-complete
"$PYTHON" -m icr.analysis --artifact-root "$ARTIFACT_ROOT" \
  > "$ARTIFACT_ROOT/logs/analysis.log" 2>&1

"$PYTHON" - "$ARTIFACT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

from icr.analysis_v2 import evidence_payload_audit
from icr.protocol import atomic_write_json

root = Path(sys.argv[1])
rows = [
    json.loads(line)
    for line in (root / "revisions" / "merged.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
audit = evidence_payload_audit(rows)
if audit is None:
    raise SystemExit("No true_evidence records found")
if audit["explicit_answer_cue_present"] or audit["source_binding_mismatches"]:
    raise SystemExit(f"Evidence payload audit failed: {audit}")
atomic_write_json(root / "analysis" / "evidence_payload_audit.json", audit)
print(json.dumps(audit, indent=2, sort_keys=True))
PY

trap - INT TERM EXIT
echo "[Evidence V1] complete: $ARTIFACT_ROOT/analysis/summary.md"
