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
PREBELIEF_SOURCE_ROOT="${PREBELIEF_SOURCE_ROOT:-}"
BOTH_CORRECT_SAMPLE="${BOTH_CORRECT_SAMPLE:-10}"

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
if [[ -n "$PREBELIEF_SOURCE_ROOT" ]]; then
  "$PYTHON" - "$PREBELIEF_SOURCE_ROOT" "$ARTIFACT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

from icr import CONDITIONS
from icr.prompts_v3 import PROMPT_VERSION
from icr.protocol import atomic_write_json, sha256_json, sha256_text

source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2])
source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
merged_text = (source / "prebeliefs" / "merged.jsonl").read_text(encoding="utf-8")
rows = [json.loads(line) for line in merged_text.splitlines() if line]
selected_ids = [int(value) for value in source_config["selected_item_ids"]]
expected = {
    (item_id, agent_id)
    for item_id in selected_ids
    for agent_id in ("A", "B")
}
actual = {(int(row["item_id"]), str(row["agent_id"])) for row in rows}
if source_config.get("dataset") != "medqa" or len(selected_ids) != 300:
    raise SystemExit("Frozen source must be the 300-item MedQA cache")
if actual != expected or len(rows) != 600:
    raise SystemExit("Frozen source prebelief cache is incomplete")
if any(row.get("status") != "complete" for row in rows):
    raise SystemExit("Frozen source contains an incomplete prebelief")

stable = {
    "protocol": "ICR-V3-RECEIVER-FROZEN-PREBELIEF-V1",
    "dataset": "medqa",
    "benchmark": "medqa300",
    "dataset_rows": int(source_config["dataset_rows"]),
    "selected_item_ids": selected_ids,
    "excluded_item_ids": [],
    "exclusion_rule": None,
    "dataset_sha256": source_config["dataset_sha256"],
    "selection": {"mode": "frozen_prebelief_source"},
    "model": source_config["model"],
    "global_seed": int(source_config["global_seed"]),
    "generation": source_config["generation"],
    "statebridge": source_config["statebridge"],
    "revision_conditions": list(CONDITIONS),
    "revision_prompt_version": PROMPT_VERSION,
    "answer_parser_version": "icr.parsing_v3@ICR-V3",
    "other_mapping_offset": int(source_config["other_mapping_offset"]),
    "replication_id": source_config.get("replication_id"),
    "prebelief_source": {
        "path": str(source),
        "protocol": source_config["protocol"],
        "config_fingerprint": source_config["fingerprint"],
        "revision_prompt_version_at_source_creation": source_config.get(
            "revision_prompt_version"
        ),
        "merged_jsonl_sha256": sha256_text(merged_text),
        "records": len(rows),
        "unique_prompt_sha256": len({row["prompt_sha256"] for row in rows}),
    },
}
candidate = {**stable, "fingerprint": sha256_json(stable)}
config_path = target / "config.json"
if config_path.exists():
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    if existing.get("fingerprint") != candidate["fingerprint"]:
        raise SystemExit("Target contains a different frozen-source configuration")
else:
    atomic_write_json(config_path, candidate)
prebelief_path = target / "prebeliefs" / "merged.jsonl"
prebelief_path.parent.mkdir(parents=True, exist_ok=True)
if prebelief_path.exists() and sha256_text(
    prebelief_path.read_text(encoding="utf-8")
) != candidate["prebelief_source"]["merged_jsonl_sha256"]:
    raise SystemExit("Target prebelief copy differs from its declared frozen source")
prebelief_path.write_text(merged_text, encoding="utf-8")
print(json.dumps(candidate["prebelief_source"], indent=2, sort_keys=True))
PY
  echo "[Evidence V1] imported frozen prebelief source; no Phase-1 generation"
elif [[ -f "$ARTIFACT_ROOT/config.json" ]] && \
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

if [[ -n "$PREBELIEF_SOURCE_ROOT" ]]; then
  "$PYTHON" - "$ARTIFACT_ROOT" > "$ARTIFACT_ROOT/logs/verification.log" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
rows = [
    json.loads(line)
    for line in (root / "prebeliefs" / "merged.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
assert config["protocol"] == "ICR-V3-RECEIVER-FROZEN-PREBELIEF-V1"
assert config["revision_prompt_version"] == "icr_v3_mid_injection"
assert len(rows) == 600
assert len({(row["item_id"], row["agent_id"]) for row in rows}) == 600
print("frozen-prebelief/latest-receiver verification: PASS")
PY
elif (( LIMIT >= 2 )); then
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

launch icr.revisions --conditions true_evidence \
  --both-correct-sample "$BOTH_CORRECT_SAMPLE" --global-resume \
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
