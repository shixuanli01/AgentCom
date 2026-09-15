#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:?usage: run_icr_revisions_4gpu.sh ARTIFACT_ROOT CONDITIONS}"
CONDITIONS="${2:-none,true_text,true_statebridge}"
PYTHON="${PYTHON:-python}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m icr.revisions --artifact-root "$ARTIFACT_ROOT" --conditions "$CONDITIONS"

"$PYTHON" -m icr.merge \
  --artifact-root "$ARTIFACT_ROOT" --phase revisions --require-complete
