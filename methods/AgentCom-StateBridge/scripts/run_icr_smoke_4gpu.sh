#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-artifacts/icr_medqa300_smoke10}"
PYTHON="${PYTHON:-python}"

PYTHON="$PYTHON" bash scripts/run_icr_prebeliefs_4gpu.sh \
  "$ARTIFACT_ROOT" --limit 10 --max-new-tokens 4096
"$PYTHON" -m icr.verify --artifact-root "$ARTIFACT_ROOT"
PYTHON="$PYTHON" bash scripts/run_icr_revisions_4gpu.sh \
  "$ARTIFACT_ROOT" \
  none,true_text,self_text,other_text,true_statebridge,self_statebridge,other_statebridge
"$PYTHON" -m icr.analysis --artifact-root "$ARTIFACT_ROOT"
