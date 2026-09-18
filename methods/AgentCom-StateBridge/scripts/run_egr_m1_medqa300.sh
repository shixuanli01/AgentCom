#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${1:-artifacts/icr_medqa300/seed_pair_00}"
OUTPUT_ROOT="${2:-artifacts/egr/medqa300_diagnostic/seed_pair_00}"
GPU="${3:-0}"

python -m egr.run_m1 \
  --source-root "$SOURCE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --gpu "$GPU" \
  --conditions claim_only evidence_only egr_zero

python -m egr.analyze_m1 \
  --source-root "$SOURCE_ROOT" \
  --egr-root "$OUTPUT_ROOT" \
  --num-bootstrap 10000 \
  --bootstrap-seed 20260916
