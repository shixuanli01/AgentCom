#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?usage: run_cross_benchmark_first_wave.sh TASK [ARTIFACT_ROOT]}"
RUN_PERMUTATION=1
case "$TASK" in
  gpqa)
    DEFAULT_ROOT="artifacts/cross_benchmark/gpqa_full_seed42"
    SELECTION_ARGS=()
    ;;
  arc_challenge)
    DEFAULT_ROOT="artifacts/cross_benchmark/arc_challenge_full_seed42"
    SELECTION_ARGS=()
    ;;
  gsm8k)
    DEFAULT_ROOT="artifacts/cross_benchmark/gsm8k_full_seed42"
    SELECTION_ARGS=()
    ;;
  mbppplus)
    DEFAULT_ROOT="artifacts/cross_benchmark/mbppplus_full_seed42"
    SELECTION_ARGS=()
    RUN_PERMUTATION=0
    ;;
  humanevalplus)
    DEFAULT_ROOT="artifacts/cross_benchmark/humanevalplus_full_seed42"
    SELECTION_ARGS=()
    RUN_PERMUTATION=0
    ;;
  *)
    echo "unsupported task: $TASK" >&2
    exit 2
    ;;
esac

if [[ -n "${BENCHMARK_SAMPLE_SIZE:-}" ]]; then
  SELECTION_ARGS=(
    --sample-size "$BENCHMARK_SAMPLE_SIZE"
    --selection-seed "${BENCHMARK_SELECTION_SEED:-42}"
  )
elif [[ -n "${BENCHMARK_LIMIT:-}" ]]; then
  SELECTION_ARGS=(--limit "$BENCHMARK_LIMIT")
fi

ARTIFACT_ROOT="${2:-$DEFAULT_ROOT}"
PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS_PER_GPU:-2}"
REPLICATION_ID="seed_pair_00"
CONDITIONS="none,true_text,true_statebridge,true_latentmas"
EGR_M1_ROOT="$ARTIFACT_ROOT/egr_m1_packets"
EGR_CONTRAST_ROOT="$ARTIFACT_ROOT/egr_contrast"
EGR_PERMUTATION_ROOT="$ARTIFACT_ROOT/egr_permutation"

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
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  pids=()
  [[ "$failed" -eq 0 ]]
}

if [[ -f "$ARTIFACT_ROOT/config.json" ]] && \
  "$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
    --phase prebeliefs --require-complete > "$ARTIFACT_ROOT/logs/prebelief_cache_check.log" 2>&1; then
  echo "[$TASK] phase 1: complete prebelief cache found; skipping generation"
else
  echo "[$TASK] phase 1: independent prebeliefs, workers=$WORKERS"
  for ((rank=0; rank<WORKERS; rank++)); do
    LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u -m icr.prebeliefs \
      --artifact-root "$ARTIFACT_ROOT" --task "$TASK" \
      --replication-id "$REPLICATION_ID" \
      "${SELECTION_ARGS[@]}" \
      --rank "$rank" --world-size "$WORKERS" \
      > "$ARTIFACT_ROOT/logs/prebelief_rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  wait_for_workers
  "$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
    --phase prebeliefs --require-complete
fi
"$PYTHON" -m icr.verify --artifact-root "$ARTIFACT_ROOT" \
  > "$ARTIFACT_ROOT/logs/verification.log" 2>&1

echo "[$TASK] phase 2: main communication baselines: $CONDITIONS"
for ((rank=0; rank<WORKERS; rank++)); do
  LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u -m icr.revisions \
    --artifact-root "$ARTIFACT_ROOT" --conditions "$CONDITIONS" \
    --global-resume --output-tag first_wave \
    --rank "$rank" --world-size "$WORKERS" \
    > "$ARTIFACT_ROOT/logs/revision_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
wait_for_workers
"$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" \
  --phase revisions --require-complete
"$PYTHON" -m icr.analysis --artifact-root "$ARTIFACT_ROOT" \
  > "$ARTIFACT_ROOT/logs/baseline_analysis.log" 2>&1

echo "[$TASK] phase 3: EGR evidence packets and contrast adjudication"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u -m egr.run_m1 \
  --source-root "$ARTIFACT_ROOT" --output-root "$EGR_M1_ROOT" \
  --conditions egr_zero --leakage-only \
  > "$ARTIFACT_ROOT/logs/egr_packets.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u -m egr.run_contrast \
  --source-root "$ARTIFACT_ROOT" --m1-root "$EGR_M1_ROOT" \
  --output-root "$EGR_CONTRAST_ROOT" --gpu 0 \
  > "$ARTIFACT_ROOT/logs/egr_contrast.log" 2>&1
EGR_ROOTS=("$EGR_CONTRAST_ROOT")
if [[ "$RUN_PERMUTATION" -eq 1 ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u -m egr.run_permutation_gate \
    --source-root "$ARTIFACT_ROOT" --m1-root "$EGR_M1_ROOT" \
    --contrast-root "$EGR_CONTRAST_ROOT" --output-root "$EGR_PERMUTATION_ROOT" \
    --gpu 0 > "$ARTIFACT_ROOT/logs/egr_permutation.log" 2>&1
  EGR_ROOTS+=("$EGR_PERMUTATION_ROOT")
fi

"$PYTHON" -m egr.analyze_cross_benchmark \
  --source-root "$ARTIFACT_ROOT" \
  --egr-roots "${EGR_ROOTS[@]}" \
  --output-root "$ARTIFACT_ROOT/cross_benchmark_analysis" \
  > "$ARTIFACT_ROOT/logs/cross_benchmark_analysis.log" 2>&1

echo "[$TASK] complete: $ARTIFACT_ROOT/cross_benchmark_analysis/report.md"
trap - INT TERM EXIT
