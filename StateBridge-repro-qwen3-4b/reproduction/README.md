# Qwen3-4B Reproduction

This directory tracks the released-code reproduction of StateBridge's five
Qwen3-4B paper results. Core upstream source files remain unchanged.

Environment:

```bash
conda activate statebridge-qwen3-4b-repro
```

Author-style single-task smoke:

```bash
python -m methods.state_bridge \
  --model Qwen/Qwen3-4B \
  --task medqa \
  --gpus 0 \
  --seed 42 \
  --limit 1
```

The five-task main track always uses the published task-specific token limits,
`K=64`, `alpha=0.3`, sequential prompts, and one GPU. Generated logs and JSON
results are ignored by Git under `reproduction/runs/`.

Audit the exact loader outputs:

```bash
python reproduction/audit_datasets.py
```

Run the five-task queue in the frozen order:

```bash
python -u reproduction/run_qwen3_4b_core.py
```

The queue refuses to silently resume a partial task because restarting the
process also restarts its sampling RNG stream. `--resume-partial` is available
for recovery, but such a result must be labeled as resumed rather than as one
uninterrupted author-style run.
