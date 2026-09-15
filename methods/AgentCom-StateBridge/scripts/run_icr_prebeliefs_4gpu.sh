#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:?usage: run_icr_prebeliefs_4gpu.sh ARTIFACT_ROOT [extra args...]}"
shift
PYTHON="${PYTHON:-python}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m icr.prebeliefs --artifact-root "$ARTIFACT_ROOT" "$@"

"$PYTHON" -m icr.merge \
  --artifact-root "$ARTIFACT_ROOT" --phase prebeliefs --require-complete
