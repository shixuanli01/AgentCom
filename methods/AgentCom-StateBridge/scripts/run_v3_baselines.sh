#!/usr/bin/env bash
# ICR-V3 full-set baseline run: none / true_text / true_statebridge / true_latentmas.
#
# Usage:
#   bash scripts/run_v3_baselines.sh TASK [ARTIFACT_ROOT]
#
# Environment:
#   CUDA_DEVICES      space-separated GPU ids (default "0 1 2 3")
#   WORKERS_PER_GPU   worker processes per GPU (default 1)
#   BENCHMARK_LIMIT   first N items only, for smoke runs
#   LATENT_STEPS      LatentMAS latent steps (default 10)
#   BOTH_CORRECT_SAMPLE  keep one in N both-correct items (default 10)
#   PHASE1_ONLY       stop after independent beliefs, before any revision
#   PYTHON            interpreter (default: repo venv)
#
# Runs are durable at item boundaries; re-running resumes completed records.
set -euo pipefail

TASK="${1:?usage: run_v3_baselines.sh TASK [ARTIFACT_ROOT]}"
case "$TASK" in
  medqa|gpqa|arc_challenge|gsm8k|mbppplus|humanevalplus) ;;
  *) echo "unsupported task: $TASK" >&2; exit 2 ;;
esac

ARTIFACT_ROOT="${2:-artifacts/icr_v3/${TASK}_full_seed42}"
DEFAULT_PYTHON=/workspace/AgentCom/.venv-agentcom/bin/python
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=python
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
REPLICATION_ID="${REPLICATION_ID:-seed_pair_00}"
CONDITIONS="none,true_text,true_statebridge,true_latentmas"
LATENT_STEPS="${LATENT_STEPS:-10}"
BOTH_CORRECT_SAMPLE="${BOTH_CORRECT_SAMPLE:-10}"
read -r -a GPUS <<< "${CUDA_DEVICES:-0 1 2 3}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
WORLD=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))

SELECTION_ARGS=()
if [[ -n "${BENCHMARK_LIMIT:-}" ]]; then
  SELECTION_ARGS=(--limit "$BENCHMARK_LIMIT")
fi

# StateBridge's vocabulary anchoring normalizes the whole embedding matrix in
# one allocation, a multi-GiB transient on top of each worker's steady state.
# Expandable segments keep that spike from fragmenting the pool when several
# workers share a GPU.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

# Fan out one process per (gpu, worker) slot. Each process sees a single GPU as
# device 0, so LOCAL_RANK stays 0 while the shard rank spans the whole world.
launch_phase() {
  local module="$1"; shift
  local rank=0 gpu worker
  for gpu in "${GPUS[@]}"; do
    for ((worker = 0; worker < WORKERS_PER_GPU; worker++)); do
      CUDA_VISIBLE_DEVICES="$gpu" LOCAL_RANK=0 "$PYTHON" -u -m "$module" \
        --artifact-root "$ARTIFACT_ROOT" \
        --rank "$rank" --world-size "$WORLD" "$@" \
        > "$ARTIFACT_ROOT/logs/${module##*.}_rank${rank}_gpu${gpu}.log" 2>&1 &
      pids+=("$!")
      rank=$((rank + 1))
    done
  done
  wait_for_workers
}

echo "[$TASK] V3 baselines | gpus=${GPUS[*]} workers/gpu=$WORKERS_PER_GPU world=$WORLD"
echo "[$TASK] artifact root: $ARTIFACT_ROOT"

if [[ -f "$ARTIFACT_ROOT/config.json" ]] && \
  "$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
    --phase prebeliefs --require-complete \
    > "$ARTIFACT_ROOT/logs/prebelief_cache_check.log" 2>&1; then
  echo "[$TASK] phase 1: complete prebelief cache found; skipping generation"
else
  echo "[$TASK] phase 1: independent prebeliefs"
  launch_phase icr.prebeliefs --task "$TASK" --replication-id "$REPLICATION_ID" \
    "${SELECTION_ARGS[@]}"
  "$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
    --phase prebeliefs --require-complete
fi

"$PYTHON" -m icr.verify --artifact-root "$ARTIFACT_ROOT" \
  > "$ARTIFACT_ROOT/logs/verification.log" 2>&1

# Phase 1 alone already tells you what phase 2 can possibly measure: CR and PR
# only exist on items where the two agents disagree about correctness, and SR
# only on items both got wrong. Running it first lets a dataset be dropped
# before its revisions are paid for.
if [[ -n "${PHASE1_ONLY:-}" ]]; then
  trap - INT TERM EXIT
  echo "[$TASK] phase 1 complete; stopping before revisions (PHASE1_ONLY)"
  exit 0
fi

echo "[$TASK] phase 2: revisions for $CONDITIONS"
launch_phase icr.revisions --conditions "$CONDITIONS" \
  --latent-steps "$LATENT_STEPS" --both-correct-sample "$BOTH_CORRECT_SAMPLE" \
  --global-resume --output-tag v3
"$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
  --phase revisions --require-complete

echo "[$TASK] phase 3: analysis"
"$PYTHON" -m icr.analysis --artifact-root "$ARTIFACT_ROOT" \
  > "$ARTIFACT_ROOT/logs/analysis.log" 2>&1
"$PYTHON" -m icr.analysis_v2 \
  --source-root "$ARTIFACT_ROOT" --seed-root "$ARTIFACT_ROOT" \
  --num-bootstrap 10000 --bootstrap-seed 20260919 \
  > "$ARTIFACT_ROOT/logs/analysis_v2.log" 2>&1

trap - INT TERM EXIT
echo "[$TASK] complete: $ARTIFACT_ROOT/analysis_v2/summary.md"
